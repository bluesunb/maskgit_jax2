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
from config import VQConfig, AutoencoderConfig, LossWeights, TrainConfig
from scripts.common import TrainState
from scripts.train_step import train_step, reconstruct_image
from utils.context import make_state, save_state, make_rngs

from datetime import datetime
from functools import partial
from numbers import Number

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


def get_now_str():
    return datetime.now().strftime("%m%d-%H%M")


def load_lpips_fn(lpips_def: LPIPS = None, img_size: int = 256, channel: int = 3):
    def identity(*args, **kwargs):
        return jp.zeros(())
    
    if lpips_def is None:
        return identity
    
    rng = jax.random.PRNGKey(0)
    lpips = LPIPS()
    params = lpips.init(rng, jp.ones((1, img_size, img_size, channel)), jp.ones((1, img_size, img_size, channel)))
    lpips = lpips.bind(params)
    return lpips.__call__


def prepare_dataset(config: TrainConfig):
    img_size = config.img_size
    batch_size = config.batch_size
    num_workers = config.num_workers

    train_transform = T.Compose([T.Resize(img_size),
                                 T.RandomCrop((img_size, img_size)),
                                 T.RandomHorizontalFlip(),
                                 T.ToTensor(),
                                 T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_transform = T.Compose([T.Resize(img_size),
                                T.CenterCrop((img_size, img_size)),
                                T.ToTensor(),
                                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    def test_untransform(x: np.ndarray):
        x = x * 0.5 + 0.5
        x = np.clip(x, 0, 1)
        x = (x * 255).astype(np.uint8)
        return x
    
    root_dir = os.path.expanduser("~/PycharmProjects/Datasets")

    if config.dataset == "imagenet":
        path = os.path.join(root_dir, "ILSVRC2012_img_test", "test")
        train_loader = load_folder_data(path,
                                        batch_size=batch_size, shuffle=True, num_workers=num_workers, transform=train_transform,
                                        max_size=config.max_size)
        
        test_loader = load_folder_data(path,
                                       batch_size=batch_size, shuffle=False, num_workers=num_workers, transform=test_transform,
                                       max_size=config.max_size)
        
    elif config.dataset == "stl":
        train_loader = load_stl(root_dir, split='train+unlabeled', shuffle=True, batch_size=batch_size, num_workers=num_workers)
        test_loader = load_stl(root_dir, split='test', shuffle=False, batch_size=batch_size, num_workers=num_workers)

    return train_loader, test_loader, test_untransform


def prepare_model(config: TrainConfig):
    img_size = config.img_size

    enc_config = AutoencoderConfig(out_channels=128,
                                   channel_multipliers=(1, 2, 2, 4))
    dec_config = AutoencoderConfig(out_channels=3,
                                   channel_multipliers=(1, 2, 2, 4))
    vq_config = VQConfig(codebook_size=512)

    gan = VQGAN(enc_config, dec_config, vq_config)
    discriminator = Discriminator(emb_channels=128, n_layers=3)

    v_rng, d_rng = jax.random.split(jax.random.PRNGKey(config.seed))

    vqgan_state = make_state(rngs=make_rngs(v_rng, ('dropout', ), contain_params=True),
                             model=gan,
                             tx=optax.chain(optax.zero_nans(), optax.adam(config.lr, b1=config.betas[0], b2=config.betas[1])),
                             inputs=jp.empty((1, img_size, img_size, 3)),
                             train=True)

    disc_state = make_state(rngs=make_rngs(d_rng, (), contain_params=True),
                            model=discriminator,
                            tx=optax.chain(optax.zero_nans(), optax.adam(config.lr, b1=config.betas[0], b2=config.betas[1])),
                            inputs=jp.empty((1, img_size, img_size, 3)),
                            train=True)
    
    lpips_fn1 = load_lpips_fn(LPIPS if config.use_lpips else None, img_size, 3)
    lpips_fn2 = load_lpips_fn(LPIPS if config.use_lpips else None, img_size, 3)
    configs = {'enc_config': enc_config, 'dec_config': dec_config, 'vq_config': vq_config}

    return vqgan_state, disc_state, lpips_fn1, lpips_fn2, configs


class Logger:
    def __init__(self):
        self.log_dict = defaultdict(list)
        self.steps = defaultdict(list)

    def log(self, info: dict, step: int):
        for k, v in info.items():
            if hasattr(v, 'ndim') and v.ndim == 0:
                self.log_dict[k].append(v.item())
                self.steps[k].append(step)
            elif isinstance(v, Number):
                self.log_dict[k].append(v)
                self.steps[k].append(step)

    def finish(self):
        pass


def main(train_config: TrainConfig,
         loss_config: LossWeights):
    
    train_loader, test_loader, test_untransform = prepare_dataset(train_config)
    vqgan_state, disc_state, lpips_fn1, lpips_fn2, vq_configs = prepare_model(train_config)

    axis_name = 'batch'
    p_train_step = jax.pmap(partial(train_step, config=loss_config, lpips_fn=lpips_fn1, pmap_axis=axis_name), axis_name=axis_name)
    p_recon_step = jax.pmap(partial(reconstruct_image, lpips_fn=lpips_fn2), axis_name=axis_name)

    vqgan_state = flax.jax_utils.replicate(vqgan_state)
    disc_state = flax.jax_utils.replicate(disc_state)
    
    def unreplicate_dict(d):
        return jax.tree_map(lambda x: jax.device_get(x[0]), d)
    
    sample_batch = next(iter(test_loader))
    if isinstance(sample_batch, (list, tuple)):
        sample_batch = sample_batch[0]
    sample_batch = shard(sample_batch[:4])
    
    # ckpt_path = os.path.abspath('./checkpoints')
    ckpt_path = os.path.join(train_config.root_dir, 'checkpoints')
    if train_config.wandb_project:
        wandb.init(project=train_config.wandb_project,
                   dir=train_config.root_dir,
                   name=f"vqgan-{get_now_str()}", 
                   config={'train_config': train_config, 'loss_config': loss_config, **vq_configs})
        
        wandb.run.config.update({'ckpt_path': ckpt_path})
        run = wandb.run
    else:
        run = Logger()

    rng = jax.random.PRNGKey(train_config.seed)
    pbar = tqdm(range(train_config.n_epochs))
    global_step = 0
    for epoch in pbar:
        for step, batch in enumerate(test_loader):
    # for epoch in range(train_config.n_epochs):
    #     pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{train_config.n_epochs}')
    #     for batch in pbar:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            
            batch = shard(batch)
            rng, device_rngs = jax.random.split(rng)
            device_rngs = shard_prng_key(device_rngs)
            (vqgan_state, disc_state), info = p_train_step(vqgan_state, disc_state, batch, device_rngs)
            # global_step = vqgan_state.step[0].item()
            global_step += 1

            if global_step % train_config.log_freq == 0:
                info = unreplicate_dict(info)
                pbar.set_postfix({'loss': info['nll_loss'] + info['g_loss'],
                                  'd_loss': info['d_loss']})

                info = flax.traverse_util.flatten_dict(info, sep='/')
                run.log(info, step=global_step)
                pbar.set_description(f'Epoch {epoch + 1}/{train_config.n_epochs} - Step {global_step}')

            if global_step % train_config.img_log_freq == 0:
                x_recon, recon_info = p_recon_step(vqgan_state, sample_batch, device_rngs)
                x_recon = jax.device_get(x_recon).squeeze()
                original = jax.device_get(sample_batch).squeeze()
                # recon_info = unreplicate_dict(recon_info)

                x_recon = np.concatenate(x_recon, axis=1)
                original = np.concatenate(original, axis=1)
                image = np.concatenate([original, x_recon], axis=0)
                image = test_untransform(image)
                image = Image.fromarray(image)

                if train_config.wandb_project:
                    run.log(recon_info, step=global_step)
                    run.log({'recon_image': wandb.Image(image)}, step=global_step)
                else:
                    image.save(f'./recon_images/{epoch}_{step}.png')
                    run.log(recon_info, step=global_step)

            if global_step % train_config.save_freq == 0 and global_step > 0:
                name = f'{epoch}_{step}.ckpt'
                save_state(vqgan_state, os.path.join(ckpt_path, 'vqgan_' + name), global_step)
                save_state(disc_state, os.path.join(ckpt_path, 'disc_' + name), global_step)

    save_state(vqgan_state, os.path.join(ckpt_path, f'vqgan_final.ckpt'), vqgan_state.step[0])
    save_state(disc_state, os.path.join(ckpt_path, f'disc_final.ckpt'), disc_state.step[0])
    print(f'Model saved ({epoch}, {step})')
    run.finish()

    # n_logs = len(run.log_dict)
    # r, c = int(n_logs ** 0.5), n_logs // int(n_logs ** 0.5)
    # fig, axes = plt.subplots(r, c, figsize=(15, 10))
    # for i, (k, v) in enumerate(run.log_dict.items()):
    #     if k == 'step':
    #         continue
    #     axes[i // c, i % c].plot(v, run.steps[k])
    #     axes[i // c, i % c].set_title(k)
    #
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    from chex import fake_pmap_and_jit
    from cProfile import Profile
    from pstats import Stats

    train_config = TrainConfig(seed=0,
                               dataset='imagenet',
                               img_size=96,
                               max_size=5 * 4,
                               batch_size=4,
                               num_workers=0,
                               n_epochs=1000,
                               log_freq=50,
                               img_log_freq=200,
                               save_freq=1000,
                               use_lpips=False,
                               lr=2.25e-5,
                               weight_decay=1e-5,
                               wandb_project="maskgit_overfit",
                               root_dir=os.path.abspath('./'))
    
    loss_config = LossWeights(log_gaussian_weight=1.0,
                              log_laplace_weight=0.0,
                              percept_weight=0.1,
                              codebook_weight=1.0,
                              adversarial_weight=0.1,
                              disc_d_start=1000,
                              disc_g_start=1000,
                              disc_flip_end=2000)
    
    # fake = False
    # if fake:
    #     with fake_pmap_and_jit():
    #         main(train_config, loss_config)
    #
    # else:
    #     main(train_config, loss_config)

    with Profile() as pr:
        main(train_config, loss_config)

    stats = Stats(pr, stream=open('profile_stats.txt', 'w'))
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats()
    stats.dump_stats('profile_stats')
    print('Profile stats saved')
