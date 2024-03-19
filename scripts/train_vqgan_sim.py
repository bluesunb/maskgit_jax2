import os, shutil
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

from utils.viz import array_to_img, plot_to_img


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
    state = checkpoints.save_checkpoint(path, state, step)
    return state


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
               lpips: LPIPS,
               config: LossWeights,
               pmap_axis: str = None):

    rng_names = {'vqgan': ('dropout', ), 'disc': ()}

    def loss_fn_nll(params):
        rngs = make_rngs(rng, rng_names['vqgan'])
        x_recon, q_loss, result = vqgan_state(batch, train=True, params=params, rngs=rngs)
        log_laplace_loss = jp.abs(x_recon - batch).mean()
        log_gaussian_loss = optax.l2_loss(x_recon, batch).mean()
        percept_loss = lpips(batch, x_recon).mean()
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
        g_loss, _ = vanilla_d_loss(logits_fake)
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

        # loss_fake = jp.mean(jax.nn.relu(1.0 + logits_fake))
        # loss_real = jp.mean(jax.nn.relu(1.0 - logits_real))
        # # disc_loss = hinge_d_loss(logits_fake, logits_real)
        # disc_loss = 0.5 * (loss_fake + loss_real)

        def positive_branch(arg):
            logits_real, logits_fake = arg
            loss_real, loss_fake = vanilla_d_loss(logits_real, logits_fake)
            return loss_real, loss_fake

        def negative_branch(arg):
            logits_real, logits_fake = arg
            loss_fake, loss_real = vanilla_d_loss(logits_fake, logits_real)
            return loss_real, loss_fake

        flip_update = jp.logical_and(disc_state.step < config.disc_d_flip, jp.mod(disc_state.step, 3) == 0)
        disc_weight = jp.asarray(disc_state.step > config.disc_d_start, dtype=jp.float32)
        loss_real, loss_fake = jax.lax.cond(flip_update, positive_branch, negative_branch, (logits_real, logits_fake))
        disc_loss = (loss_real + loss_fake) * 0.5 * disc_weight
        return disc_loss, (updates, {'loss_real': loss_real, 'loss_fake': loss_fake})

    (nll_loss, result), nll_grads = jax.value_and_grad(loss_fn_nll, has_aux=True)(vqgan_state.params)
    g_loss, g_grads = jax.value_and_grad(loss_fn_gen)(vqgan_state.params)
    (d_loss, (updates, d_result)), d_grads = jax.value_and_grad(loss_fn_disc, has_aux=True)(disc_state.params)

    result.update(d_result)

    last_layer_nll = jp.linalg.norm(nll_grads['decoder']['ConvOut']['kernel'])
    last_layer_gen = jp.linalg.norm(g_grads['decoder']['ConvOut']['kernel'])

    disc_weight_factor = jp.where(vqgan_state.step < config.disc_g_start, config.adversarial_weight, 0.0)
    disc_weight = last_layer_nll / (last_layer_gen + 1e-4) * disc_weight_factor
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

    vqgan_state = vqgan_state.apply_gradients(nll_grads)
    vqgan_state = vqgan_state.apply_gradients(g_grads)
    disc_state = disc_state.apply_gradients(d_grads)
    disc_state = disc_state.replace(extra_variables=updates)

    result.update({'nll_loss': nll_loss, 'g_loss': g_loss, 'd_loss': d_loss, 'disc_weight': disc_weight})

    return (vqgan_state, disc_state), result


def reconstruct_image(vqgan_state, disc_state, batch, rng, lpips):
    rng_names = {'vqgan': ('dropout', ), 'disc': ()}
    x_recon, q_loss, result = vqgan_state(batch, train=False, rngs=make_rngs(rng, rng_names['vqgan']))
    percept_loss = lpips(x_recon, batch).mean()
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

    # train_loader = load_folder_data(os.path.expanduser("~/PycharmProjects/Datasets/ILSVRC2012_img_test/test"),
    #                                 batch_size=batch_size, shuffle=True, num_workers=num_workers, transform=train_transform)
    # test_loader = load_folder_data(os.path.expanduser("~/PycharmProjects/Datasets/ILSVRC2012_img_test/test"),
    #                                batch_size=batch_size, shuffle=False, num_workers=num_workers, transform=test_transform)

    train_loader = load_stl(os.path.expanduser('~/PycharmProjects/Datasets'), 'train+unlabeled',
                            batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_loader = load_stl(os.path.expanduser('~/PycharmProjects/Datasets'), 'test',
                           batch_size=batch_size, shuffle=False, num_workers=num_workers)

    sample_batch = next(iter(test_loader))
    sample_batch = shard(sample_batch[0][0:4])

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

    lpips1 = LPIPS()
    lpips1 = lpips1.bind(
        lpips1.init(rng, jp.ones((1, img_size, img_size, 3)), jp.ones((1, img_size, img_size, 3)))
    )

    lpips2 = LPIPS()
    lpips2 = lpips2.bind(
        lpips2.init(rng, jp.ones((1, img_size, img_size, 3)), jp.ones((1, img_size, img_size, 3)))
    )

    parallel_train_step = jax.pmap(partial(train_step, lpips=lpips1, config=loss_config, pmap_axis='batch'), axis_name='batch')
    parallel_recon_step = jax.pmap(partial(reconstruct_image, lpips=lpips2))

    vqgan_state = flax.jax_utils.replicate(vqgan_state)
    disc_state = flax.jax_utils.replicate(disc_state)
    unreplicate_dict = lambda x: jax.tree_map(lambda y: jax.device_get(y[0]), x)

    wandb.init(project='maskgit', dir=os.path.abspath('./wandb'), name=f'maskgit_{get_now_str()}')
    run = wandb.run
    ckpt_path = os.path.abspath('./checkpoints/')

    for epoch in range(n_epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs}")
        for step, batch in enumerate(pbar):
            batch = shard(batch[0])
            rng, device_rngs = jax.random.split(rng)
            device_rngs = shard_prng_key(device_rngs)
            (vqgan_state, disc_state), info = parallel_train_step(vqgan_state, disc_state, batch, device_rngs)

            info = unreplicate_dict(info)
            pbar.set_postfix({'vq_loss': info['nll_loss'] + info['g_loss'],
                              'disc_loss': info['d_loss'],
                              'disc_weight': info['disc_weight']})

            global_step = epoch * len(train_loader) + step
            run.log({k: v.item() for k, v in info.items() if v.ndim == 0},
                    step=epoch * len(train_loader) + step)

            if global_step % 1500 == 0 and step > 0:
                # Save model
                save_state(vqgan_state, os.path.join(ckpt_path, f'vqgan_{epoch}_{step}.ckpt'), vqgan_state.step[0])
                save_state(disc_state, os.path.join(ckpt_path, f'disc_{epoch}_{step}.ckpt'), disc_state.step[0])
                print(f'Model saved ({epoch}, {step})')

            if global_step % 250 == 0:
                # Save image
                x_recon, recon_info = parallel_recon_step(vqgan_state, disc_state, sample_batch, device_rngs)
                x_recon = jax.device_get(x_recon[0]).squeeze()
                original = jax.device_get(sample_batch[0][0]).squeeze()
                recon_info = unreplicate_dict(recon_info)
                run.log({'reconstructions': wandb.Image(np.concatenate([original, x_recon], axis=1)),
                         **recon_info},
                        step=global_step)

    save_state(vqgan_state, os.path.join(ckpt_path, f'vqgan_{epoch}_{step}.ckpt'), vqgan_state.step[0])
    save_state(disc_state, os.path.join(ckpt_path, f'disc_{epoch}_{step}.ckpt'), disc_state.step[0])
    print(f'Model saved ({epoch}, {step})')
    run.finish()


if __name__ == "__main__":
    from chex import fake_pmap_and_jit
    from cProfile import Profile
    from pstats import Stats

    rng = jax.random.PRNGKey(0)
    disc_factor = 1.0
    disc_start = 10000
    percept_loss_weight = 1.0
    recon_loss_weight = 1.0
    loss_config = LossWeights(disc_d_start=3000,
                              disc_g_start=3000,
                              disc_d_flip=6000)

    # with fake_pmap_and_jit():
    #     main(rng, img_size=96, batch_size=2, num_workers=8, n_epochs=1, loss_config=loss_config)
    # main(rng, img_size=96, batch_size=2, num_workers=8, n_epochs=1, loss_config=loss_config)

    with Profile() as pr:
        main(rng, img_size=96, batch_size=128, num_workers=8, n_epochs=50, loss_config=loss_config)

    stats = Stats(pr, stream=open('profile_stats.txt', 'w'))
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats()
    stats.dump_stats('profile_stats.pstat')
    print('Done')