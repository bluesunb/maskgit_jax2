from dataclasses import dataclass
from typing import Tuple


@dataclass
class VQConfig:
    codebook_size: int = 1024
    commit_loss_weight: float = 0.25
    entropy_loss_weight: float = 0.1
    entropy_temperature: float = 0.01
    # entroy_loss_type: str = "softmax"
    # quantizer_type: str = "vq"


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

    # n_img_tokens: int = 24      # number of img tokens = seq_len
    codebook_size: int = 1024   # codebook tokens = VQConfig.codebook_size
    sample_temperature: float = 4.5
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
    # recon_loss: float = 1.0
    codebook_weight: float = 1.0
    # disc_factor: int = 1

    adversarial_weight: float = 0.1
    disc_g_start: int = 10000
    disc_d_start: int = 10000
    disc_flip_end: int = 20000


@dataclass
class TrainConfig:
    seed: int = 0
    dataset: str = "imagenet"
    img_size: int = 256
    max_size: int = 100000
    batch_size: int = 8
    num_workers: int = 8

    n_epochs: int = 50
    log_freq: int = 10
    img_log_freq: int = 250
    save_freq: int = 1000
    use_lpips: bool = True

    lr: float = 1e-4
    betas: Tuple[float, float] = (0.5, 0.9)
    weight_decay: float = 1e-4
    grad_accum: int = 1

    wandb_project: str = ""
    root_dir: str = ""


if __name__ == "__main__":
    import flax.linen as nn
    import jax, jax.numpy as jp


    class ResidualMLPBlock(nn.Module):
        @nn.compact
        def __call__(self, x, _):
            h = nn.Dense(features=256)(x)
            h = nn.relu(h)
            return x + h, None


    class ResidualMLP(nn.Module):
        n_layers: int = 4

        @nn.compact
        def __call__(self, x):
            ScanMLP = nn.scan(
                ResidualMLPBlock, variable_axes={'params': 0},
                variable_broadcast=False, split_rngs={'params': True},
                length=self.n_layers)
            x, _ = ScanMLP()(x, None)
            return x

    class ResidualMLP2(nn.Module):
        n_layers: int = 4

        @nn.compact
        def __call__(self, x):
            y = nn.Dense(features=256)(x)
            for _ in range(self.n_layers):
                h = nn.Dense(features=256)(x)
                h = nn.relu(h) + y
                x = x + h
            return x


    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (1, 4, 256))
    model1 = ResidualMLP(200)
    model2 = ResidualMLP2(200)

    from time import time

    st = time()
    param1 = model1.init(rng, x)
    print(f'init time: {time() - st:.4f}')
    st = time()
    param2 = model2.init(rng, x)
    print(f'init time: {time() - st:.4f}')

    a1 = jax.jit(model1.apply)
    a2 = jax.jit(model2.apply)

    # y1 = model1.apply(param1, x)
    # y2 = model2.apply(param2, x)

    print(jp.allclose(y1, y2))