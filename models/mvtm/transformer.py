import os
import math
import numpy as np
import jax, jax.numpy as jp
import flax.linen as nn
import optax

from models.vqgan.vqvae import VQGAN
from models.mvtm.bi_transformer import BidirectionalTransformer, BidirectionalTransformer2
from config import TransformerConfig
from scripts.common import TrainState
from utils.context import load_state


def get_mask_schedule(mode="cosine"):
    if mode == "linear":
        return lambda r: 1 - r
    elif mode == "cosine":
        return lambda r: np.cos(r * np.pi / 2)
    elif mode == "square":
        return lambda r: 1 - r ** 2
    elif mode == "cubic":
        return lambda r: 1 - r ** 3

    raise NotImplementedError(f"Mask schedule mode {mode} not implemented")


class VQGANTransformer(nn.Module):
    config: TransformerConfig
    vqgan: VQGAN

    def setup(self):
        self.sos_token = self.config.codebook_size + 1
        self.mask_token_id = self.config.codebook_size
        self.scheduler = get_mask_schedule(self.config.mask_scheme)

    def encode_to_z(self, x: jp.ndarray, train: bool = True):
        x_enc = self.vqgan.encode(x)
        quantized, q_loss, result = self.vqgan.quantize(x_enc)
        indices = result["indices"]
        quantized = jax.lax.stop_gradient(quantized)
        indices = jax.lax.stop_gradient(indices)
        return quantized, indices

    @nn.compact
    def __call__(self, x: jp.ndarray, train: bool = True):
        quantized, z_indices = self.encode_to_z(x, False)        # z_indices: [bs, h/, w/, z_dim]
        z_indices = z_indices.reshape((x.shape[0], -1))
        sos_tokens = jp.ones((x.shape[0], 1), dtype=jp.int32) * self.sos_token

        mask_rng = self.make_rng('mask')
        n_mask_rng, mask_sample_rng = jax.random.split(mask_rng)
        n_indices = z_indices.shape[1]
        n_masks = math.floor(jax.random.uniform(n_mask_rng, ()) * n_indices)
        sample = jax.random.uniform(mask_sample_rng, z_indices.shape)
        sample = (-sample).argsort(1)[:, :n_masks]
        sample = jax.nn.one_hot(sample, n_indices).any(axis=1)      # where to leave the token (0: will masked)
        mask = jp.where(sample, True, False)

        masked = mask * z_indices + (1 - mask) * self.mask_token_id
        masked = jp.concatenate([sos_tokens, masked], axis=1)   # [sos, z1, m2, ..., mn]
        logits = BidirectionalTransformer(self.config, n_indices)(masked, train)
        target = jp.concatenate([sos_tokens, z_indices], axis=1)    # [sos, z1, z2, ..., zn]
        return logits, target


if __name__ == "__main__":
    from pprint import pp
    from config import AutoencoderConfig, VQConfig
    config = TransformerConfig(emb_dim=128,
                               n_heads=8,
                               n_layers=3,
                               intermediate_dim=4 * 128,
                               attn_pdrop=0.1,
                               resid_pdrop=0.1,
                               ff_pdrop=0.1,
                               # n_tokens=24,
                               codebook_size=240,
                               temperature=4.5,
                               mask_scheme="cosine")

    vqgan = VQGAN(enc_config=AutoencoderConfig(out_channels=128),
                  dec_config=AutoencoderConfig(out_channels=3),
                  vq_config=VQConfig(codebook_size=config.codebook_size),
                  train=False)

    rng = jax.random.PRNGKey(0)
    vq_param = vqgan.init(rng, jp.empty((2, 96, 96, 3)))
    vqgan = vqgan.bind(vq_param)

    model = VQGANTransformer(config, vqgan)
    print(vqgan(jp.ones((2, 96, 96, 3)))[0].shape)

    rng, mask_rng = jax.random.split(rng)
    x = jax.random.normal(rng, (2, 96, 96, 3))
    params = model.init({'params': rng, 'mask': mask_rng, 'dropout': rng}, x, train=True)

    def loss_fn(params, x):
        logits, target = model.apply(params, x, train=True, rngs={'mask': mask_rng, 'dropout': rng})
        return optax.softmax_cross_entropy_with_integer_labels(logits, target).mean()

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(params, x)
    print(loss)

    pp(jax.tree_map(lambda x: jp.linalg.norm(x).item(), grad), width=180)