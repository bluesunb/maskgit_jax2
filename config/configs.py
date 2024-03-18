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
    emb_dim: int = 768
    n_heads: int = 12
    n_layers: int = 24
    intermediate_dim: int = 4 * 768
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    ff_pdrop: float = 0.1


@dataclass
class AutoencoderConfig:
    channels: int = 128
    out_channels: int = 3
    channel_multipliers: Tuple[int, ...] = (1, 1, 2, 2, 4)
    attn_resolutions: Tuple[int, ...] = (16, )
    n_blocks: int = 2
    dropout_rate: float = 0.0
    resample_with_conv: bool = True


@dataclass
class LossWeights:
    log_gaussian_loss: float = 1.0
    log_laplace_loss: float = 0.0
    percept_loss: float = 0.1
    recon_loss: float = 1.0
    codebook_loss: float = 1.0
    disc_factor: int = 1
    disc_start: int = 10000
    disc_gan_start: int = 10000
    adversarial_weight: float = 1.0