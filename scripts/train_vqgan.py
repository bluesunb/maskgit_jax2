import os, shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import jax
import flax
from flax.training.common_utils import shard, shard_prng_key
from tqdm import tqdm

import wandb

from utils.train_prepare import prepare_dataset, prepare_vqgan
from config import LossWeights, TrainConfig
from scripts.train_step import train_step, reconstruct_image
from utils.context import save_state, unreplicate_dict, Logger

from datetime import datetime
from functools import partial

import pickle
from pprint import pprint
from PIL import Image
import matplotlib.pyplot as plt


def imshow(img):
    plt.axis('off')
    plt.imshow(img);
    plt.show()


def traced_imshow(batch, recons):
    plt.axis('off')
    orig = jax.device_get(batch[0].val[0])
    recon = jax.device_get(recons[0].primal.val[0])
    plt.imshow(np.concatenate([orig, recon], axis=1))
    plt.show()


def get_now_str():
    return datetime.now().strftime("%m%d-%H%M")


def main(train_config: TrainConfig,
         loss_config: LossWeights):
    train_loader, test_loader, test_untransform = prepare_dataset(train_config)
    vqgan_state, disc_state, lpips_fn1, lpips_fn2, vq_configs = prepare_vqgan(train_config)

    axis_name = 'batch'
    p_train_step = jax.pmap(partial(train_step, config=loss_config, lpips_fn=lpips_fn1, pmap_axis=axis_name),
                            axis_name=axis_name)
    p_recon_step = jax.pmap(partial(reconstruct_image, lpips_fn=lpips_fn2), axis_name=axis_name)

    vqgan_state = flax.jax_utils.replicate(vqgan_state)
    disc_state = flax.jax_utils.replicate(disc_state)

    sample_batch = next(iter(test_loader))
    if isinstance(sample_batch, (list, tuple)):
        sample_batch = sample_batch[0]
    sample_batch = shard(sample_batch[:4])

    plt.imshow(np.concatenate(test_untransform(jax.device_get(sample_batch).squeeze()), axis=1))
    plt.show()

    root_dir = Path(train_config.root_dir).absolute()
    save_path: Path = root_dir / 'saves'
    if train_config.wandb_project:
        wandb.init(project=train_config.wandb_project,
                   dir=train_config.root_dir,
                   name=f"vqgan-{get_now_str()}",
                   config={'train_config': train_config, 'loss_config': loss_config, **vq_configs})

        wandb.run.config.update({'ckpt_path': save_path})
        run = wandb.run
    else:
        run = Logger()

    rng = jax.random.PRNGKey(train_config.seed)
    global_step = 0
    pbar = tqdm(range(train_config.n_epochs))
    for epoch in pbar:
        for step, batch in enumerate(test_loader):
    # for epoch in range(train_config.n_epochs):
    #     pbar = tqdm(enumerate(train_loader), desc=f'Epoch {epoch + 1}/{train_config.n_epochs}', total=len(train_loader))
    #     for step, batch in pbar:
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
                    # if not os.path.exists('./recon_images'):
                    #     os.makedirs('./recon_images', exist_ok=True)
                    name = f'{str(epoch).zfill(4)}_{str(step).zfill(4)}.png'
                    # image.save(f'./recon_images/{name}')
                    image.save(save_path / 'recon_images' / name)
                    run.log(recon_info, step=global_step)

            if global_step % train_config.save_freq == 0 and global_step > 0:
                name = f'{str(epoch).zfill(4)}_{str(step).zfill(4)}.ckpt'
                save_state(vqgan_state, save_path / f'vqgan_{name}', global_step)
                save_state(disc_state, save_path / f'disc_{name}', global_step)

    save_state(vqgan_state, save_path / 'vqgan_final.ckpt', vqgan_state.step[0])
    save_state(disc_state, save_path / 'disc_final.ckpt', disc_state.step[0])
    pickle.dump(vq_configs, open(save_path / 'vq_configs.pkl', 'wb'))
    disc_config = disc_state.model_def.__dict__
    disc_config = {k: v for k, v in disc_config.items() if not k.startswith('_')}
    pickle.dump(disc_config, open(save_path / 'disc_configs.pkl', 'wb'))
    print(f'Model saved ({epoch}, {step})')

    vqgan_state = flax.jax_utils.unreplicate(vqgan_state)
    pickle.dump({'params': vqgan_state.params, **vqgan_state.extra_variables},
                open(save_path / 'vqgan_params.pkl', 'wb'))
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

    root_dir = Path(os.path.abspath('./'))
    if os.path.exists(os.path.join(root_dir, 'train_step.py')):
        root_dir = root_dir.parent

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
                               wandb_project="",
                               root_dir=str(root_dir))

    loss_config = LossWeights(log_gaussian_weight=1.0,
                              log_laplace_weight=0.0,
                              percept_weight=0.1,
                              codebook_weight=1.0,
                              adversarial_weight=0.1,
                              disc_d_start=1000,
                              disc_g_start=1000,
                              disc_flip_end=2000)

    fake = False
    if fake:
        with fake_pmap_and_jit():
            main(train_config, loss_config)

    else:
        main(train_config, loss_config)

    # with Profile() as pr:
    #     main(train_config, loss_config)
    #
    # stats = Stats(pr, stream=open('profile_stats.txt', 'w'))
    # stats.strip_dirs()
    # stats.sort_stats('cumulative')
    # stats.print_stats()
    # stats.dump_stats('profile_stats')
    # print('Profile stats saved')
