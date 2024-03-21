from dataclasses import dataclass
from typing import Tuple


@dataclass
class VQConfig:
    # out_channels: int
    # n_res_blocks: int
    # channel_multipliers: int
    # emb_dim: int
    # conv_downsample: bool
    # norm_type: str
    # act_fn: str

    codebook_size: int = 1024
    commit_loss_weight: float = 0.25
    entropy_loss_weight: float = 0.1
    # entroy_loss_type: str = "softmax"
    entropy_temperature: float = 0.01
    # quantizer_type: str


@dataclass
class TransformerConfig:
    # emb_dim: int = 768
    emb_dim: int = 128
    # n_heads: int = 12
    n_heads: int = 12
    # n_layers: int = 24
    n_layers: int = 3
    # intermediate_dim: int = 4 * 768
    intermediate_dim: int = 4 * 128
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    ff_pdrop: float = 0.1

    n_tokens: int = 24      # number of img tokens = seq_len
    codebook_size: int = 1024   # codebook tokens = VQConfig.codebook_size
    temperature: float = 4.5
    mask_scheme: str = "cosine"


@dataclass
class AutoencoderConfig:
    channels: int = 64
    out_channels: int = 3
    # channel_multipliers: Tuple[int, ...] = (1, 1, 2, 2, 4)
    # channel_multipliers: Tuple[int, ...] = (1, 1, 2, 2, 2)
    channel_multipliers: Tuple[int, ...] = (1,1,2,2,4,)
    attn_resolutions: Tuple[int, ...] = (24, )
    n_blocks: int = 2
    dropout_rate: float = 0.1
    resample_with_conv: bool = True


@dataclass
class LossWeights:
    log_gaussian_weight: float = 1.0
    log_laplace_weight: float = 0.0
    percept_weight: float = 0.1
    adversarial_weight: float = 0.1
    # recon_loss: float = 1.0
    codebook_loss: float = 1.0
    # disc_factor: int = 1

    disc_g_start: int = 10000
    disc_d_start: int = 10000
    disc_d_flip: int = 20000
