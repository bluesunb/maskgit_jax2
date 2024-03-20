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
        for i in range(2):
            c = self.emb_channels * min(2 ** (self.idx + i), 8)
            x = DiscriminatorLayer(c, down=(i == 1))(x)
        
        c = self.emb_channels * min(2 ** (self.idx + 1), 8)
        residual = nn.Conv(c, (3, 3), (2, 2), use_bias=False)(residual)
        x = (x + residual) * jp.sqrt(0.5)
        return x


class Discriminator(nn.Module):
    emb_channels: int = 64
    kernel_size: int = 4
    n_layers: int = 3

    @nn.compact
    def __call__(self, x: jp.ndarray, train: bool = True):
        x = nn.Conv(self.emb_channels, (1, 1), padding='SAME')(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        
        n_layers = min(jp.log2(x.shape[1]).astype(int), self.n_layers)

        for i in range(self.n_layers):
            c = self.emb_channels * min(2 ** i, 8)
            s = 2 if i < self.n_layers else 1
            # x = nn.Conv(c, (4, 4), (s, s), padding=1, use_bias=False)(x)
            x = DiscriminatorBlock(self.emb_channels, idx=i)(x)
            # x = nn.BatchNorm(momentum=0.9, epsilon=1e-5, axis=(0, -1))(x, use_running_average=False)
            # x = nn.BatchNorm()(x, use_running_average=not train)
            x = instance_norm(x)
            x = nn.leaky_relu(x, negative_slope=0.2)

        x = nn.Conv(1, (4, 4), (1, 1), padding='SAME')(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(1)(x)
        return x

# class Discriminator(nn.Module):
#     emb_channels: int = 64
#     # n_layers: int = 3

#     @nn.compact
#     def __call__(self, x: jp.ndarray, train: bool = True):
#         x = nn.Conv(self.emb_channels, (1, 1), padding='SAME')(x)
#         x = nn.leaky_relu(x, negative_slope=0.2)

#         n_layers = jp.log2(x.shape[1]).astype(int)

#         for i in range(n_layers):
#             c = self.emb_channels * min(2 ** i, 8)
#             x = nn.Conv(c, (1, 1), use_bias=False, padding='SAME')(x)
#             x = DiscriminatorBlock(c, idx=i)(x)
#             x = instance_norm(x)
#             x = nn.leaky_relu(x, negative_slope=0.2)

#         x = nn.Conv(1, (1, 1), padding='SAME')(x)
#         return x


if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    x = jax.random.normal(rng, (1, 256, 256, 3))
    model = Discriminator()
    params = model.init(init_rng, x)
    y = model.apply(params, x, train=False)
    print(y.shape)