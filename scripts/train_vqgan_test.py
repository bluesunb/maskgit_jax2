import os, shutil

import chex
import numpy as np
import jax, jax.numpy as jp
import flax, optax
from flax.training import checkpoints
from flax.training.common_utils import shard, shard_prng_key
from tqdm import tqdm

import wandb
import torchvision.transforms as T

from models.vqgan import VQGAN, Discriminator
from utils.metrics import LPIPS
from utils.dataset import load_folder_data, load_stl
from config import VQConfig, AutoencoderConfig, LossWeights
from scripts.common import TrainState

from datetime import datetime
from functools import partial
from typing import Sequence, Union, Dict, Callable

from pprint import pprint
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict


def imshow(img):
    plt.axis('off')
    plt.imshow(img);plt.show()


def traced_imshow(batch, recons):
    plt.axis('off')
    orig = jax.device_get(batch[0].val[0])
    recon = jax.device_get(recons[0].primal.val[0])
    plt.imshow(np.concatenate([orig, recon], axis=1))
    plt.show()


def merge_dict(d1, d2):
    for k, v in d2.items():
        d1[k].append(v)
    return d1


def get_now_str():
    return datetime.now().strftime("%m%d-%H%M")


def make_rngs(rng: jp.ndarray,
              names: Union[Sequence[str], Dict[str, Sequence[int]]] = None,
              contain_params: bool = False):

    if names is None:
        return jax.random.split(rng)[1]

    if isinstance(names, Sequence):
        rng, *rngs = jax.random.split(rng, len(names) + 1)
        rngs = {name: r for name, r in zip(names, rngs)}
        if contain_params:
            rngs['params'] = rng
        return rngs

    rngs = jax.random.split(rng, len(names))
    return {name: make_rngs(r, names[name], contain_params) for name, r in zip(names, rngs)}


def make_state(rngs, model, tx, init_batch_shape):
    inputs = jp.empty(init_batch_shape)
    variables = jax.jit(model.init, static_argnames=['train'])(rngs, inputs, train=True)
    # variables = model.init(rngs, inputs, train=True)
    state = TrainState.create(model,
                              params=variables.pop('params'),
                              tx=tx,
                              extra_variables=variables)

    return state


def save_state(state: TrainState, path: str, step: int):
    if os.path.exists(path):
        shutil.rmtree(path)
    state = flax.jax_utils.unreplicate(state)
    state = jax.device_get(state)
    checkpoints.save_checkpoint(path, target=state, step=step, overwrite=True, keep=2)


def vanilla_d_loss(logits_fake, logits_real = None):
    if logits_real is None:
        loss_fake = jp.mean(jax.nn.softplus(-logits_fake)) * 2
        loss_real = 0
    else:
        loss_fake = jp.mean(jax.nn.softplus(logits_fake))
        loss_real = jp.mean(jax.nn.softplus(-logits_real))

    # return (loss_fake + loss_real) * 0.5
    return loss_fake, loss_real


def hinge_d_loss(logits_fake, logits_real = None):
    loss_fake = jp.mean(jax.nn.relu(1.0 + logits_fake))
    if logits_real is None:
        loss_real = 0
    else:
        loss_real = jp.mean(jax.nn.relu(1.0 - logits_real))

    return (loss_fake + loss_real) * 0.5


def train_step(vqgan_state: TrainState,
               disc_state: TrainState,
               batch: jp.ndarray,
               rng: jp.ndarray,
               # lpips: LPIPS,
               config: LossWeights,
               pmap_axis: str = None):

    rng_names = {'vqgan': ('dropout', ), 'disc': ()}

    def loss_fn_nll(params):
        rngs = make_rngs(rng, rng_names['vqgan'])
        x_recon, q_loss, result = vqgan_state(batch, train=True, params=params, rngs=rngs)
        log_laplace_loss = jp.abs(x_recon - batch).mean()
        log_gaussian_loss = optax.l2_loss(x_recon, batch).mean()
        # percept_loss = lpips(batch, x_recon).mean()
        percept_loss = 0
        nll_loss = (log_laplace_loss * config.log_laplace_weight
                    + log_gaussian_loss * config.log_gaussian_weight
                    + percept_loss * config.percept_weight)

        nll_loss = nll_loss + q_loss * config.codebook_loss
        result.update({'percept_loss': percept_loss, 'q_loss': q_loss})
        return nll_loss, result

    def loss_fn_gen(params):
        rngs_vq = make_rngs(rng, rng_names['vqgan'])
        rngs_disc = make_rngs(rng, rng_names['disc'])
        x_recon, q_loss, result = vqgan_state(batch, train=True, params=params, rngs=rngs_vq)
        logits_fake = disc_state(x_recon, train=False, rngs=rngs_disc)
        g_loss = jp.mean(jax.nn.softplus(-logits_fake)) * 2
        return g_loss * 0.5

    def loss_fn_disc(params):
        rngs_vq = make_rngs(rng, rng_names['vqgan'])
        rngs_disc = make_rngs(rng, rng_names['disc'])

        x_recon, q_loss, result = vqgan_state(batch, train=False, rngs=rngs_vq)
        x_recon = jax.lax.stop_gradient(x_recon)

        logits_fake, updates = disc_state(x_recon, train=True, rngs=rngs_disc,
                                          params=params, mutable=['batch_stats'])
        logits_real, updates = disc_state(batch, train=True, rngs=rngs_disc,
                                          params=params, extra_variables=updates, mutable=['batch_stats'])

        def positive_branch(arg):   # loss for generator
            logits_real, logits_fake = arg
            # loss_real, loss_fake = vanilla_d_loss(logits_real, logits_fake)
            loss_real = jp.mean(jax.nn.softplus(logits_real))
            loss_fake = jp.mean(jax.nn.softplus(-logits_fake))
            return loss_real, loss_fake

        def negative_branch(arg):   # loss for discriminator
            logits_real, logits_fake = arg
            # loss_fake, loss_real = vanilla_d_loss(logits_fake, logits_real)
            loss_real = jp.mean(jax.nn.softplus(-logits_real))
            loss_fake = jp.mean(jax.nn.softplus(logits_fake))
            return loss_real, loss_fake

        flip_update = jp.logical_and(disc_state.step < config.disc_d_flip, jp.mod(disc_state.step, 3) == 0)
        loss_real, loss_fake = jax.lax.cond(flip_update, positive_branch, negative_branch, (logits_real, logits_fake))
        # disc_weight = jp.asarray(disc_state.step > config.disc_d_start, dtype=jp.float32)
        disc_weight = jp.where(disc_state.step > config.disc_d_start, 1.0, 0.0)
        disc_loss = (loss_real + loss_fake) * 0.5 * disc_weight
        return disc_loss, (updates, {'loss_real': loss_real, 'loss_fake': loss_fake, 'flip_update': flip_update, 'disc_weight': disc_weight})

    (nll_loss, result), nll_grads = jax.value_and_grad(loss_fn_nll, has_aux=True)(vqgan_state.params)
    g_loss, g_grads = jax.value_and_grad(loss_fn_gen)(vqgan_state.params)
    (d_loss, (updates, d_result)), d_grads = jax.value_and_grad(loss_fn_disc, has_aux=True)(disc_state.params)

    result.update(d_result)

    # last_layer_nll = jp.linalg.norm(nll_grads['decoder']['ConvOut']['kernel'])
    # last_layer_gen = jp.linalg.norm(g_grads['decoder']['ConvOut']['kernel'])

    disc_weight_factor = jp.where(vqgan_state.step > config.disc_g_start, config.adversarial_weight, 0.0)
    # disc_weight = last_layer_nll / (last_layer_gen + 1e-4) * disc_weight_factor
    disc_weight = disc_weight_factor
    disc_weight = jp.clip(disc_weight, 0, 1e4)

    g_grads = jax.tree_map(lambda x: x * disc_weight, g_grads)

    if pmap_axis is not None:
        nll_grads = jax.lax.pmean(nll_grads, axis_name=pmap_axis)
        g_grads = jax.lax.pmean(g_grads, axis_name=pmap_axis)
        d_grads = jax.lax.pmean(d_grads, axis_name=pmap_axis)

        nll_loss = jax.lax.pmean(nll_loss, axis_name=pmap_axis)
        g_loss = jax.lax.pmean(g_loss, axis_name=pmap_axis)
        d_loss = jax.lax.pmean(d_loss, axis_name=pmap_axis)

        result = jax.lax.pmean(result, axis_name=pmap_axis)

    g_grads = jax.tree_map(lambda x, y: (x + y) * 0.5, g_grads, nll_grads)
    # vqgan_state = vqgan_state.apply_gradients(nll_grads)
    vqgan_state = vqgan_state.apply_gradients(g_grads)
    disc_state = disc_state.apply_gradients(d_grads)
    disc_state = disc_state.replace(extra_variables=updates)

    result.update({'nll_loss': nll_loss, 'g_loss': g_loss, 'd_loss': d_loss, 'g_grad_factor': disc_weight, 'gen_weight': disc_weight_factor})

    return (vqgan_state, disc_state), result


def reconstruct_image(vqgan_state, disc_state, batch, rng):
    rng_names = {'vqgan': ('dropout', ), 'disc': ()}
    x_recon, q_loss, result = vqgan_state(batch, train=False, rngs=make_rngs(rng, rng_names['vqgan']))
    # percept_loss = lpips(x_recon, batch).mean()
    percept_loss = 0
    recon_loss = optax.l2_loss(x_recon, batch).mean()

    logits_fake = disc_state(x_recon, train=False, rngs=make_rngs(rng, rng_names['disc']))
    loss_fake = jp.mean(jax.nn.relu(1.0 + logits_fake))

    return x_recon, {'recon_loss': recon_loss, 'percept_loss': percept_loss, 'loss_fake': loss_fake}


def main(rng,
         img_size: int = 256,
         batch_size: int = 8,
         num_workers: int = 8,
         n_epochs: int = 50,
         loss_config: LossWeights = LossWeights()):

    # Load dataset
    train_transform = T.Compose([T.Resize(img_size),
                                 T.RandomCrop((img_size, img_size)),
                                 T.RandomHorizontalFlip(),
                                 T.ToTensor()])

    test_transform = T.Compose([T.Resize(img_size),
                                T.CenterCrop((img_size, img_size)),
                                T.ToTensor()])

    train_loader = load_folder_data(os.path.expanduser("~/PycharmProjects/Datasets/ILSVRC2012_img_test/test"),
                                    batch_size=batch_size, shuffle=True, num_workers=num_workers, transform=train_transform,
                                    max_size=2 * batch_size)
    test_loader = load_folder_data(os.path.expanduser("~/PycharmProjects/Datasets/ILSVRC2012_img_test/test"),
                                   batch_size=batch_size, shuffle=False, num_workers=num_workers, transform=test_transform,
                                   max_size=2 * batch_size)

    # train_loader = load_stl(os.path.expanduser('~/PycharmProjects/Datasets'), 'train+unlabeled',
    #                         batch_size=batch_size, shuffle=True, num_workers=num_workers)
    #
    # test_loader = load_stl(os.path.expanduser('~/PycharmProjects/Datasets'), 'test',
    #                        batch_size=batch_size, shuffle=False, num_workers=num_workers)

    sample_batch = next(iter(test_loader))
    if isinstance(sample_batch, (list, tuple)):
        sample_batch = sample_batch[0]
    sample_batch = shard(sample_batch[0:4])

    enc_config = AutoencoderConfig(out_channels=256)
    dec_config = AutoencoderConfig(out_channels=3)
    vq_config = VQConfig(codebook_size=1024)

    gan = VQGAN(enc_config, dec_config, vq_config)
    discriminator = Discriminator(emb_channels=64, n_layers=3)
    # trainer_model = VQGANTrainer(gan, discriminator)
    # state = make_state(rngs=make_rngs(rng, rng_names, contain_params=True),
    #                    model=trainer_model,
    #                    tx=optax.chain(optax.zero_nans(), optax.adam(2.25e-5)),
    #                    init_batch_shape=(1, img_size, img_size, 3))

    vqgan_state = make_state(rngs=make_rngs(rng, ('dropout', ), contain_params=True),
                             model=gan,
                             tx=optax.chain(optax.zero_nans(), optax.adam(2.25e-5)),
                             init_batch_shape=(1, img_size, img_size, 3))

    disc_state = make_state(rngs=make_rngs(rng, contain_params=True),
                            model=discriminator,
                            tx=optax.chain(optax.zero_nans(), optax.adam(2.25e-5)),
                            init_batch_shape=(1, img_size, img_size, 3))

    # lpips1 = LPIPS()
    # lpips1 = lpips1.bind(
    #     lpips1.init(rng, jp.ones((1, img_size, img_size, 3)), jp.ones((1, img_size, img_size, 3)))
    # )
    #
    # lpips2 = LPIPS()
    # lpips2 = lpips2.bind(
    #     lpips2.init(rng, jp.ones((1, img_size, img_size, 3)), jp.ones((1, img_size, img_size, 3)))
    # )

    # parallel_train_step = jax.pmap(partial(train_step, lpips=lpips1, config=loss_config, pmap_axis='batch'), axis_name='batch')
    parallel_train_step = jax.pmap(partial(train_step, config=loss_config, pmap_axis='batch'), axis_name='batch')
    # parallel_recon_step = jax.pmap(partial(reconstruct_image, lpips=lpips2))
    parallel_recon_step = jax.pmap(reconstruct_image)

    vqgan_state = flax.jax_utils.replicate(vqgan_state)
    disc_state = flax.jax_utils.replicate(disc_state)
    unreplicate_dict = lambda x: jax.tree_map(lambda y: jax.device_get(y[0]), x)

    # wandb.init(project='maskgit', dir=os.path.abspath('./wandb'), name=f'maskgit_{get_now_str()}')
    # run = wandb.run
    # ckpt_path = os.path.abspath('./checkpoints/')

    log = defaultdict(list)
    pbar = tqdm(range(n_epochs), desc=f"Epoch 0/{n_epochs}")
    # for epoch in range(n_epochs):
    for epoch in pbar:
        # pbar = tqdm(test_loader, desc=f"Epoch {epoch}/{n_epochs}")
        for step, batch in enumerate(test_loader):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = shard(batch)
            rng, device_rngs = jax.random.split(rng)
            device_rngs = shard_prng_key(device_rngs)
            (vqgan_state, disc_state), info = parallel_train_step(vqgan_state, disc_state, batch, device_rngs)
            global_step = epoch * len(test_loader) + step

            if global_step % 50 == 0:
                info = unreplicate_dict(info)
                pbar.set_postfix({'vq_loss': info['nll_loss'] + info['g_loss'],
                                  'disc_loss': info['d_loss'],
                                  'g_loss': info['g_loss'],
                                  'd_loss': info['d_loss']})

                # print(f'Epoch {epoch}/{n_epochs}, step {step}/{len(train_loader)}: step: {global_step}')
                # pprint({'vq_loss': info['nll_loss'] + info['g_loss'],
                #         'disc_loss': info['d_loss'],
                #         'disc_weight': info['disc_weight']})

                log = merge_dict(log, {k: v.item() for k, v in info.items() if v.ndim == 0})
                log['step'].append(vqgan_state.step[0].item())
                pbar.set_description(f'Epoch {epoch}/{n_epochs}, step {disc_state.step[0]}/{len(train_loader)} ({vqgan_state.step[0]}):')

            # run.log({k: v.item() for k, v in info.items() if v.ndim == 0},
            #         step=epoch * len(train_loader) + step)
            # run.log({'step': vqgan_state.step[0]}, step=global_step)

            # if global_step % 1500 == 0 and step > 0:
            #     # Save model
            #     save_state(vqgan_state, os.path.join(ckpt_path, f'vqgan_last.ckpt'), vqgan_state.step[0])
            #     save_state(disc_state, os.path.join(ckpt_path, f'disc_last.ckpt'), disc_state.step[0])
            #     print(f'Model saved ({epoch}, {step})')

            # if global_step % 250 == 0:
            #     Save image
            if global_step % 150 == 0:
                x_recon, recon_info = parallel_recon_step(vqgan_state, disc_state, sample_batch, device_rngs)
                x_recon = jax.device_get(x_recon).squeeze()
                original = jax.device_get(sample_batch).squeeze()
                # recon_info = unreplicate_dict(recon_info)

                x_recon = np.concatenate(x_recon, 1)
                original = np.concatenate(original, 1)
                image = np.concatenate([original, x_recon], 0)
                image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image)
                image.save(f'./image/{epoch}_{step}.png')
    #             run.log({'reconstructions': wandb.Image(np.concatenate([original, x_recon], axis=0)),
    #                      **recon_info},
    #                     step=global_step)
    #
    # save_state(vqgan_state, os.path.join(ckpt_path, f'vqgan_final.ckpt'), vqgan_state.step[0])
    # save_state(disc_state, os.path.join(ckpt_path, f'disc_final.ckpt'), disc_state.step[0])
    # print(f'Model saved ({epoch}, {step})')
    # run.finish()

    import pandas
    log_df = pandas.DataFrame(log)
    fig, axes = plt.subplots(3, 5, figsize=(15, 10))
    for i, col in enumerate(log_df.columns):
        axes[i//5, i%5].plot(log_df[col])
        axes[i//5, i%5].set_title(col)
    # log_df['step'].plot()
    # log_df['nll_loss'].plot()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from chex import fake_pmap_and_jit
    from cProfile import Profile
    from pstats import Stats

    rng = jax.random.PRNGKey(0)
    disc_factor = 1.0
    disc_start = 10000
    percept_loss_weight = 1.0
    recon_loss_weight = 1.0
    loss_config = LossWeights(disc_d_start=1500,
                              disc_g_start=1500,
                              disc_d_flip=0,
                              adversarial_weight=0.05)

    # with chex.fake_pmap():
    #     main(rng, img_size=96, batch_size=8, num_workers=8, n_epochs=50, loss_config=loss_config)
    # main(rng, img_size=96, batch_size=2, num_workers=8, n_epochs=1, loss_config=loss_config)

    # with Profile() as pr:
    # with chex.fake_pmap_and_jit():
    main(rng, img_size=96, batch_size=4, num_workers=0, n_epochs=2000, loss_config=loss_config)

    stats = Stats(pr, stream=open('profile_stats.txt', 'w'))
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats()
    stats.dump_stats('profile_stats.pstat')
    print('Done')