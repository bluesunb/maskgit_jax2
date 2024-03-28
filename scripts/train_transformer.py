import os
from pathlib import Path
from datetime import datetime

import numpy as np
import jax, jax.numpy as jp
import flax, optax
from flax.training.common_utils import shard, shard_prng_key

import pickle
import wandb
from torchvision import transforms as T
from tqdm import tqdm
from PIL import Image

from config import TransformerConfig, TrainConfig
from models.mvtm.transformer import VQGANTransformer
from scripts.common import TrainState
from utils.context import make_rngs, unreplicate_dict, save_state, Logger
from utils.dataset import load_folder_data, load_stl
from utils.train_prepare import prepare_dataset, prepare_transformer


def get_now_str():
    return datetime.now().strftime("%m%d-%H%M")


def train_step(trns_state: TrainState, batch: jp.ndarray, rng: jp.ndarray, pmap_axis: str = None):
    rng_names = ('mask', 'dropout')
    rngs = make_rngs(rng, rng_names)

    def loss_fn(params):
        logits, target = trns_state(batch, train=True, params=params, rngs=rngs)
        return optax.softmax_cross_entropy_with_integer_labels(logits, target).mean()
    
    loss, grad = jax.value_and_grad(loss_fn)(trns_state.params)
    grad = jax.lax.pmean(grad, axis_name=pmap_axis)
    trns_state = trns_state.apply_gradients(grad)
    return trns_state, loss


def main(train_config: TrainConfig):
    ckpt_path = Path(os.path.join(train_config.root_dir, 'checkpoints'))
    train_loader, test_loader, test_untransform = prepare_dataset(train_config)
    transformer, trns_config = prepare_transformer(vq_path=ckpt_path / 'vqgan_configs.pkl')

    axis_name = 'batch'
    p_train_step = jax.pmap(train_step, axis_name=axis_name)

    sample_batch = next(iter(test_loader))
    if isinstance(sample_batch, (list, tuple)):
        sample_batch = sample_batch[0]
    sample_batch = shard(sample_batch[:4])

    ckpt_path = Path(os.path.join(train_config.root_dir, 'checkpoints'))
    if train_config.wandb_project:
        wandb.init(project=train_config.wandb_project,
                   dir=train_config.root_dir,
                   name=f"vqgan-{get_now_str()}",
                   config={'train_config': train_config, **trns_config})
        wandb.run.config.update({'ckpt_path': ckpt_path})
        run = wandb.run
    else:
        run = Logger()

    rng = jax.random.PRNGKey(train_config.seed)
    global_step = 0

    for epoch in range(train_config.n_epochs):
        pbar = tqdm(enumerate(train_loader), desc=f'Epoch {epoch + 1}/{train_config.n_epochs}', total=len(train_loader))
        for step, batch in pbar:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]

            batch = shard(batch)
            rng, device_rngs = jax.random.split(rng)
            device_rngs = shard_prng_key(device_rngs)
            trns_state, loss = p_train_step(trns_state, batch, device_rngs)
            global_step += 1
            pbar.set_postfix({'loss': loss})

        x_recon, x_sample, x_new = trns_state(sample_batch, method='log_images')
        image = np.concatenate([jax.device_get(x_new),
                                jax.device_get(x_sample),
                                jax.device_get(x_recon)], axis=0)
        image = test_untransform(image)
        image = Image.fromarray(image)
        
        if train_config.wandb_project:
            run.log({'recon_images': wandb.Image(image)}, step=global_step)
        else:
            image.save('./recon_images/trns_{epoch}.png')
        
        if train_config.save_freq == 0:
            name = f'trns_{epoch}.ckpt'
            save_state(trns_state, ckpt_path / name, global_step)
        
    save_state(trns_state, ckpt_path / 'trns_final.ckpt', global_step)
    pickle.dump(trns_config, open(ckpt_path / 'trns_configs.pkl', 'wb'))
    print(f"Model saved at {ckpt_path}")
    run.finish()
