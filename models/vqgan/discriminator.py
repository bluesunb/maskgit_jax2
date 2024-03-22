import math
import jax
import jax.numpy as jp
import flax
import flax.linen as nn
from functools import partial

    
InstanceNorm = partial(nn.BatchNorm, momentum=0.9, epsilon=1e-5, axis=(0, 1,))


def instance_norm(x, epsilon=1e-5):
    mean = jp.mean(x, axis=(2, 3), keepdims=True)
    variance = jp.var(x, axis=(2, 3), keepdims=True)
    inv_std = jp.reciprocal(jp.sqrt(variance + epsilon))
    normalized_x = (x - mean) * inv_std
    return normalized_x


class DiscriminatorLayer(nn.Module):
    out_channels: int
    use_bias: bool = False
    down: bool = False

    @nn.compact
    def __call__(self, x: jp.ndarray):
        s = 2 if self.down else 1
        x = nn.Conv(self.out_channels, (3, 3), strides=(s, s), use_bias=False)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        return x
    

class DiscriminatorBlock(nn.Module):
    emb_channels: int = 64
    idx: int = 0

    @nn.compact
    def __call__(self, x: jp.ndarray):
        residual = x
        c = self.emb_channels * min(2 ** self.idx, 8)
        x = DiscriminatorLayer(c, down=True)(x)
        x = DiscriminatorLayer(c, down=False)(x)
        residual = nn.Conv(c, (3, 3), (2, 2), use_bias=False)(residual)
        x = (x + residual) * jp.sqrt(0.5)

        x = instance_norm(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        return x


class Discriminator(nn.Module):
    emb_channels: int = 64
    kernel_size: int = 4
    n_layers: int = 3

    def setup(self):
        self.block = nn.Sequential([DiscriminatorBlock(self.emb_channels, idx=i) for i in range(self.n_layers)])

    @nn.compact
    def __call__(self, x: jp.ndarray, train: bool = True):
        x = nn.Conv(self.emb_channels, (1, 1), padding='SAME')(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        x = self.block(x)
        x = nn.Conv(1, (4, 4), (1, 1), padding='SAME')(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(1)(x)
        return x


if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    x = jax.random.normal(rng, (1, 256, 256, 3))
    model1 = Discriminator(n_layers=5)
    # model2 = Discriminator2(n_layers=5)

    param1 = jax.jit(model1.init)({'params': init_rng, 'dropout': init_rng}, x, True)
    # param2 = jax.jit(model2.init)({'params': init_rng, 'dropout': init_rng}, x, True)

    a1 = jax.jit(model1.apply)
    # a2 = jax.jit(model2.apply)

    y1 = a1(param1, x)
    # y2 = a2(param2, x)

    print(y1.shape, y2.shape)