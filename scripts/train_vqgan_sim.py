import os
import numpy as np
import jax
import jax.numpy as jp
import flax
import flax.linen as nn
import optax
from flax.training import checkpoints
from flax.training.common_utils import shard, shard_prng_key
from tqdm import tqdm

import torchvision.transforms as T

from models.vqgan import VQGAN, Discriminator
from utils.metrics import LPIPS
from utils.dataset import load_data
from config import VQConfig, AutoencoderConfig, LossWeights
from scripts.common import TrainState

from functools import partial
from typing import Sequence, Union, Dict, Callable


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


def vanilla_d_loss(logits_fake, logits_real = None):
    if logits_real is None:
        loss_fake = jp.mean(jax.nn.softplus(-logits_fake)) * 2
        loss_real = 0
    else:
        loss_fake = jp.mean(jax.nn.softplus(logits_fake))
        loss_real = jp.mean(jax.nn.softplus(-logits_real))

    return (loss_fake + loss_real) * 0.5


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
        percept_loss = lpips(x_recon, batch).mean()
        recon_loss = optax.l2_loss(x_recon, batch).mean()
        nll_loss = config.percept_loss * percept_loss + config.recon_loss * recon_loss + q_loss
        result.update({'percept_loss': percept_loss, 'recon_loss': recon_loss})
        return nll_loss + q_loss, result

    def loss_fn_gen(params):
        rngs_vq = make_rngs(rng, rng_names['vqgan'])
        rngs_disc = make_rngs(rng, rng_names['disc'])
        x_recon, q_loss, result = vqgan_state(batch, train=True, params=params, rngs=rngs_vq)
        logits_fake = disc_state(x_recon, train=False, rngs=rngs_disc)
        g_loss = -jp.mean(logits_fake)
        return g_loss

    def loss_fn_disc(params):
        rngs_vq = make_rngs(rng, rng_names['vqgan'])
        rngs_disc = make_rngs(rng, rng_names['disc'])

        x_recon, q_loss, result = vqgan_state(batch, train=False, rngs=rngs_vq)
        x_recon = jax.lax.stop_gradient(x_recon)

        logits_fake, updates = disc_state(x_recon, train=True, rngs=rngs_disc,
                                          params=params, mutable=['batch_stats'])
        logits_real, updates = disc_state(batch, train=True, rngs=rngs_disc,
                                          params=params, extra_variables=updates, mutable=['batch_stats'])

        loss_fake = jp.mean(jax.nn.relu(1.0 + logits_fake))
        loss_real = jp.mean(jax.nn.relu(1.0 - logits_real))
        # disc_loss = hinge_d_loss(logits_fake, logits_real)
        disc_loss = 0.5 * (loss_fake + loss_real)

        disc_factor = jp.where(disc_state.step < config.disc_start, 0.0, config.disc_factor)
        return disc_loss * disc_factor, (updates, {'loss_fake': loss_fake, 'loss_real': loss_real})

    (nll_loss, result), nll_grads = jax.value_and_grad(loss_fn_nll, has_aux=True)(vqgan_state.params)
    g_loss, g_grads = jax.value_and_grad(loss_fn_gen)(vqgan_state.params)
    (d_loss, (updates, d_result)), d_grads = jax.value_and_grad(loss_fn_disc, has_aux=True)(disc_state.params)

    result.update(d_result)

    if pmap_axis is not None:
        nll_grads = jax.lax.pmean(nll_grads, axis_name=pmap_axis)
        g_grads = jax.lax.pmean(g_grads, axis_name=pmap_axis)
        d_grads = jax.lax.pmean(d_grads, axis_name=pmap_axis)

        nll_loss = jax.lax.pmean(nll_loss, axis_name=pmap_axis)
        g_loss = jax.lax.pmean(g_loss, axis_name=pmap_axis)
        d_loss = jax.lax.pmean(d_loss, axis_name=pmap_axis)

        result = jax.lax.pmean(result, axis_name=pmap_axis)

    last_layer_nll = jp.linalg.norm(nll_grads['decoder']['ConvOut']['kernel'])
    last_layer_gen = jp.linalg.norm(g_grads['decoder']['ConvOut']['kernel'])
    disc_weight = last_layer_nll / (last_layer_gen + 1e-4)
    disc_weight = jp.clip(disc_weight, 0, 1e4)

    g_grads = jax.tree_map(lambda x: x * disc_weight, g_grads)

    vqgan_state = vqgan_state.apply_gradients(nll_grads)
    vqgan_state = vqgan_state.apply_gradients(g_grads)
    disc_state = disc_state.apply_gradients(d_grads)
    disc_state = disc_state.replace(extra_variables=updates)

    result.update({'nll_loss': nll_loss, 'g_loss': g_loss, 'd_loss': d_loss, 'disc_weight': disc_weight})

    return (vqgan_state, disc_state), result


def main(rng,
         disc_factor=1.0,
         disc_start=10000,
         percept_loss_weight=1.0,
         recon_loss_weight=1.0, ):
    img_size = 256
    batch_size = 1
    num_workers = 8

    enc_config = AutoencoderConfig(out_channels=256)
    dec_config = AutoencoderConfig(out_channels=3)
    vq_config = VQConfig(codebook_size=1024)

    # rng_names = {'vqgan': ('dropout', ),
    #              'disc': ()}
    rng_names = ('dropout',)

    # Prepare model
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

    lpips = LPIPS()
    lpips = lpips.bind(
        lpips.init(rng, jp.ones((1, img_size, img_size, 3)), jp.ones((1, img_size, img_size, 3)))
    )

    # Load dataset
    train_transform = T.Compose([T.Resize(img_size),
                                 T.RandomCrop((img_size, img_size)),
                                 T.RandomHorizontalFlip(),
                                 T.ToTensor()])

    test_transform = T.Compose([T.Resize(img_size),
                                T.CenterCrop((img_size, img_size)),
                                T.ToTensor()])

    train_loader = load_data(os.path.expanduser("~/PycharmProjects/Datasets/ILSVRC2012_img_test/test"),
                             batch_size=batch_size, shuffle=True, num_workers=num_workers, transform=train_transform)
    test_loader = load_data(os.path.expanduser("~/PycharmProjects/Datasets/ILSVRC2012_img_test/test"),
                            batch_size=batch_size, shuffle=False, num_workers=num_workers, transform=test_transform)

    n_epochs = 50
    n_steps = len(train_loader)

    loss_config = LossWeights()
    parallel_train_step = jax.pmap(partial(train_step, lpips=lpips, config=loss_config, pmap_axis='batch'), axis_name='batch')

    vqgan_state = flax.jax_utils.replicate(vqgan_state)
    disc_state = flax.jax_utils.replicate(disc_state)
    unreplicate_dict = lambda x: jax.tree_map(lambda y: jax.device_get(y[0]), x)

    for epoch in range(n_epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs}")
        for step, batch in enumerate(pbar):
            batch = shard(batch)
            rng, device_rngs = jax.random.split(rng)
            device_rngs = shard_prng_key(device_rngs)
            (vqgan_state, disc_state), info = parallel_train_step(vqgan_state, disc_state, batch, device_rngs)

            if step % 1000 == 0:
                # Save model
                pass

            if step % 100 == 0:
                # Save image
                pass


if __name__ == "__main__":
    from chex import fake_pmap_and_jit

    rng = jax.random.PRNGKey(0)
    disc_factor = 1.0
    disc_start = 10000
    percept_loss_weight = 1.0
    recon_loss_weight = 1.0

    with fake_pmap_and_jit():
        main(rng, disc_factor, disc_start, percept_loss_weight, recon_loss_weight)
    # main(rng, disc_factor, disc_start, percept_loss_weight, recon_loss_weight)