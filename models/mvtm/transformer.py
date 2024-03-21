import os
import math
import numpy as np
import jax, jax.numpy as jp
import flax.linen as nn
from models.vqgan.vqvae import VQGAN
from models.mvtm.bi_transformer import BidirectionalTransformer
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
    path: str
    vqgan: VQGAN
    config: TransformerConfig

    def setup(self):
        self.sos_token = self.config.codebook_size + 1
        self.mask_token_id = self.config.codebook_sizes
        self.transformer = BidirectionalTransformer(self.config)
        self.scheduler = get_mask_schedule(self.config.mask_scheme)

    def encode_to_z(self, x: jp.ndarray, train: bool = True):
        x_enc = self.vqgan.encode(x, train)
        quantized, q_loss, result = self.vqgan.quantize(x_enc, train)
        indices = result["indices"]
        quantized = jax.lax.stop_gradient(quantized)
        indices = jax.lax.stop_gradient(indices)
        return quantized, indices

    def __call__(self, x: jp.ndarray, train: bool = True):
        quantized, z_indices = self.encode_to_z(x, train)
        sos_tokens = jp.ones((x.shape[0], 1), dtype=jp.int32) * self.sos_token

        n_masks = math.floor()