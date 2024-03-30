import os
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import jax, jax.numpy as jp
import flax, optax, chex
from flax.training.common_utils import shard, shard_prng_key

import pickle
import wandb
from torchvision import transforms as T
from tqdm import tqdm
from PIL import Image
from functools import partial

from config import TransformerConfig, TrainConfig
from scripts.common import TrainState
from utils.context import make_rngs, unreplicate_dict, save_state, Logger
from utils.train_prepare import prepare_dataset, prepare_transformer
from utils.viz import array_to_heatmap, plot_to_array


def get_now_str():
    return datetime.now().strftime("%m%d-%H%M")


def train_step(trns_state: TrainState, batch: jp.ndarray, rng: jp.ndarray, pmap_axis: str = None):
    rng_names = ('mask', 'dropout')
    rngs = make_rngs(rng, rng_names)

    def loss_fn(params):
        logits, target = trns_state(batch, train=True, params=params, rngs=rngs)
        # loss_img = optax.softmax_cross_entropy_with_logits(logits, target)  # (bs, seq_len)
        loss_img = optax.softmax_cross_entropy_with_integer_labels(logits, target)  # (bs, seq_len)
        return loss_img.mean(), loss_img
    
    (loss, loss_img), grad = jax.value_and_grad(loss_fn, has_aux=True)(trns_state.params)
    grad = jax.lax.pmean(grad, axis_name=pmap_axis)
    loss = jax.lax.pmean(loss, axis_name=pmap_axis)
    loss_img = jax.lax.pmean(loss_img, axis_name=pmap_axis)
    trns_state = trns_state.apply_gradients(grad)
    return trns_state, (loss, loss_img)


def main(train_config: TrainConfig):
    root_dir = Path(train_config.root_dir).absolute()
    save_path = root_dir / 'saves'

    train_loader, test_loader, test_untransform = prepare_dataset(train_config)
    trns_state, trns_config = prepare_transformer(train_config,
                                                  vq_param_path=save_path / 'vqgan_params.pkl',
                                                  vq_config_path=save_path / 'vq_configs.pkl')

    axis_name = 'batch'
    p_train_step = jax.pmap(partial(train_step, pmap_axis=axis_name), axis_name=axis_name)
    p_log_images = jax.pmap(partial(trns_state.__call__, method='log_images'), axis_name=axis_name)
    trns_state = flax.jax_utils.replicate(trns_state)

    sample_batch = next(iter(test_loader))
    if isinstance(sample_batch, (list, tuple)):
        sample_batch = sample_batch[0]
    # sample_batch = sample_batch[:1]
    sample_batch = shard(sample_batch[:4])

    if train_config.wandb_project:
        wandb.init(project=train_config.wandb_project,
                   dir=train_config.root_dir,
                   name=f"vqgan-{get_now_str()}",
                   config={'train_config': train_config, **trns_config})
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
        # pbar = tqdm(enumerate(train_loader), desc=f'Epoch {epoch + 1}/{train_config.n_epochs}', total=len(train_loader))
        # for step, batch in pbar:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]

            batch = shard(batch)
            rng, device_rngs = jax.random.split(rng)
            device_rngs = shard_prng_key(device_rngs)
            trns_state, (loss, loss_img) = p_train_step(trns_state, batch, device_rngs)
            global_step += 1
            pbar.set_postfix({'loss': loss[0]})

            run.log(unreplicate_dict({'loss': loss}), global_step)

        if epoch % train_config.img_log_freq == 0:
            x_recon, x_sample, x_new = p_log_images(sample_batch, rngs={'fill': shard_prng_key(rng)})
            # x_recon, x_sample, x_new = flax.jax_utils.unreplicate(trns_state)(sample_batch[0], method='log_images', rngs={'fill': rng})
            image = np.concatenate([jax.device_get(x_new[0, 0]),
                                    jax.device_get(x_sample[0, 0]),
                                    jax.device_get(x_recon[0, 0])], axis=-2)
            image = test_untransform(image)
            image = Image.fromarray(image)

            if train_config.wandb_project:
                run.log({'recon_images': wandb.Image(image)}, step=global_step)
                array_to_heatmap(run, 'loss_img', jax.device_get(loss_img[0, 0]), step=global_step)
            else:
                # image.save('./recon_images/trns_{epoch}.png')
                name = f'{str(epoch).zfill(4)}.png'
                image.save(save_path / 'recon_images' / f'trns_recon_{name}')

                fig, axes = plt.subplots()
                plt.pcolormesh(jax.device_get(loss_img.mean(axis=(0, 1), keepdims=True)[0]))
                img = plot_to_array(fig)
                img = Image.fromarray(img.astype(np.uint8))
                img.save(save_path / 'recon_images' / f'trns_loss_{name}')

        if epoch % train_config.save_freq == 0:
            name = f'trns_{epoch}.ckpt'
            save_state(trns_state, save_path / name, global_step)

    save_state(trns_state, save_path / 'trns_final.ckpt', global_step)
    pickle.dump(trns_config, open(save_path / 'trns_configs.pkl', 'wb'))
    print(f"Model saved at {save_path}")
    run.finish()

    plt.plot(run.log_dict['loss'])
    plt.show()


if __name__ == '__main__':
    root_dir = Path(os.path.abspath('./'))
    if os.path.exists(os.path.join(root_dir, 'train_step.py')):
        root_dir = root_dir.parent

    train_config = TrainConfig(seed=0,
                               dataset='imagenet',
                               img_size=128,
                               max_size=5 * 4,
                               batch_size=4,
                               num_workers=0,
                               n_epochs=200,
                               log_freq=1,
                               img_log_freq=50,
                               save_freq=100,
                               use_lpips=False,
                               lr=1e-4,
                               grad_accum=10,
                               # root_dir=Path('./maskgit_jax2/scripts').absolute())
                               root_dir=str(root_dir))

    # import chex
    # with chex.fake_pmap_and_jit():
    #     main(train_config)
    main(train_config)