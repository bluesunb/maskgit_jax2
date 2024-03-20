import os
import math
import numpy as np
import jax, jax.numpy as jp
import flax.linen as nn
from bi_transformer import BidirectionalTransformer
from config import TransformerConfig


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

    def setup(self):
        self.sos_token = self.config.codebook_size + 1
        self.mask_token_id = self.config.codebook_sizes

    