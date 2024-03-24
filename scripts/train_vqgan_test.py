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
from scripts.train_step import train_step, reconstruct_image, make_rngs

from datetime import datetime
from functools import partial

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


def load_lpips_fn(lpips_def: LPIPS = None, img_size: int = 256, channel: int = 3):
    def identity(*args, **kwargs):
        return 0
    
    if lpips_def is None:
        return identity
    
    rng = jax.random.PRNGKey(0)
    lpips = LPIPS()
    params = lpips.init(rng, jp.ones((1, img_size, img_size, channel)), jp.ones((1, img_size, img_size, channel)))
    lpips = lpips.bind(params)
    return lpips.__call__


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
                                    max_size=400 * batch_size)
    test_loader = load_folder_data(os.path.expanduser("~/PycharmProjects/Datasets/ILSVRC2012_img_test/test"),
                                   batch_size=batch_size, shuffle=False, num_workers=num_workers, transform=test_transform,
                                   max_size=400 * batch_size)

    # train_loader = load_stl(os.path.expanduser('~/PycharmProjects/Datasets'), 'train+unlabeled',
    #                         batch_size=batch_size, shuffle=True, num_workers=num_workers)
    #
    # test_loader = load_stl(os.path.expanduser('~/PycharmProjects/Datasets'), 'test',
    #                        batch_size=batch_size, shuffle=False, num_workers=num_workers)

    sample_batch = next(iter(test_loader))
    if isinstance(sample_batch, (list, tuple)):
        sample_batch = sample_batch[0]
    sample_batch = shard(sample_batch[0:4])

    enc_config = AutoencoderConfig(out_channels=128,
                                   channel_multipliers=(1, 2, 2, 4))
    dec_config = AutoencoderConfig(out_channels=3,
                                   channel_multipliers=(1, 2, 2, 4))
    vq_config = VQConfig(codebook_size=512)

    gan = VQGAN(enc_config, dec_config, vq_config)
    discriminator = Discriminator(emb_channels=128, n_layers=3)

    vqgan_state = make_state(rngs=make_rngs(rng, ('dropout', ), contain_params=True),
                             model=gan,
                             tx=optax.chain(optax.zero_nans(), optax.adam(4e-5)),
                             init_batch_shape=(1, img_size, img_size, 3))

    disc_state = make_state(rngs=make_rngs(rng, contain_params=True),
                            model=discriminator,
                            tx=optax.chain(optax.zero_nans(), optax.adam(4e-5)),
                            init_batch_shape=(1, img_size, img_size, 3))
    
    lpips_fn = load_lpips_fn()

    # parallel_train_step = jax.pmap(partial(train_step, lpips=lpips1, config=loss_config, pmap_axis='batch'), axis_name='batch')
    parallel_train_step = jax.pmap(partial(train_step, config=loss_config, pmap_axis='batch'), axis_name='batch')
    # parallel_recon_step = jax.pmap(partial(reconstruct_image, lpips=lpips2))
    parallel_recon_step = jax.pmap(reconstruct_image)

    vqgan_state = flax.jax_utils.replicate(vqgan_state)
    disc_state = flax.jax_utils.replicate(disc_state)
    unreplicate_dict = lambda x: jax.tree_map(lambda y: jax.device_get(y[0]), x)

    wandb.init(project='maskgit', dir=os.path.abspath('./wandb'), name=f'maskgit_{get_now_str()}')
    run = wandb.run
    ckpt_path = os.path.abspath('./checkpoints/')

    log = defaultdict(list)
    # pbar = tqdm(range(n_epochs), desc=f"Epoch 0/{n_epochs}")
    for epoch in range(n_epochs):
    # for epoch in pbar:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs}")
        # for step, batch in enumerate(train_loader):
        for batch in pbar:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = shard(batch)
            rng, device_rngs = jax.random.split(rng)
            device_rngs = shard_prng_key(device_rngs)
            (vqgan_state, disc_state), info = parallel_train_step(vqgan_state, disc_state, batch, device_rngs)
            # global_step = epoch * len(train_loader) + step
            global_step = vqgan_state.step[0].item()

            if global_step % 10 == 0:
                info = unreplicate_dict(info)
                pbar.set_postfix({'vq_loss': info['nll_loss'] + info['g_loss'],
                                  'disc_loss': info['d_loss'],
                                  'g_loss': info['g_loss'],
                                  'd_loss': info['d_loss']})

            run.log({k: v.item() for k, v in info.items() if v.ndim == 0},
                    step=global_step)

            # log = merge_dict(log, {k: v.item() for k, v in info.items() if v.ndim == 0})
            # log['step'].append(vqgan_state.step[0].item())
            # pbar.set_description(f'Epoch {epoch}/{n_epochs}, step {step}/{len(train_loader)} ({vqgan_state.step[0]}):')
            pbar.set_description(f'Epoch {epoch}/{n_epochs}, step {global_step}:')

                # print(f'Epoch {epoch}/{n_epochs}, step {step}/{len(train_loader)}: step: {global_step}')
                # pprint({'vq_loss': info['nll_loss'] + info['g_loss'],
                #         'disc_loss': info['d_loss'],
                #         'disc_weight': info['disc_weight']})


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
            if global_step % 200 == 0:
                x_recon, recon_info = parallel_recon_step(vqgan_state, disc_state, sample_batch, device_rngs)
                x_recon = jax.device_get(x_recon).squeeze()
                original = jax.device_get(sample_batch).squeeze()
                recon_info = unreplicate_dict(recon_info)

                x_recon = np.concatenate(x_recon, 1)
                original = np.concatenate(original, 1)
                image = np.concatenate([original, x_recon], 0)
                image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image)
                image.save(f'./image/{epoch}_{global_step % len(train_loader)}.png')
                run.log({'reconstructions': wandb.Image(image), global_step: global_step})
                run.log({f'recon/{k}': v.item() for k, v in recon_info.items() if v.ndim == 0}, global_step=global_step)

    save_state(vqgan_state, os.path.join(ckpt_path, f'vqgan_final.ckpt'), vqgan_state.step[0])
    save_state(disc_state, os.path.join(ckpt_path, f'disc_final.ckpt'), disc_state.step[0])
    print(f'Model saved ({epoch}, {global_step % len(train_loader)})')
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
    loss_config = LossWeights(disc_d_start=5000,
                              disc_g_start=5000,
                              disc_flip_end=6500,
                              adversarial_weight=0.1)

    # with chex.fake_pmap():
    #     main(rng, img_size=96, batch_size=8, num_workers=8, n_epochs=50, loss_config=loss_config)
    # main(rng, img_size=96, batch_size=2, num_workers=8, n_epochs=1, loss_config=loss_config)

    # with Profile() as pr:
    # with chex.fake_pmap_and_jit():
    main(rng, img_size=96, batch_size=140, num_workers=10, n_epochs=50, loss_config=loss_config)

    stats = Stats(pr, stream=open('profile_stats.txt', 'w'))
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats()
    stats.dump_stats('profile_stats.pstat')
    print('Done')