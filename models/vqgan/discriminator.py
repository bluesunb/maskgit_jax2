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


class Discriminator(nn.Module):
    emb_channels: int = 64
    n_layers: int = 3

    @nn.compact
    def __call__(self, x: jp.ndarray, train: bool = True):
        x = nn.Conv(self.emb_channels, (4, 4), (2, 2), padding=1)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        
        for i in range(1, self.n_layers + 1):
            c = self.emb_channels * min(2 ** i, 8)
            s = 2 if i < self.n_layers else 1
            x = nn.Conv(c, (4, 4), (s, s), padding=1, use_bias=False)(x)
            # x = nn.BatchNorm(momentum=0.9, epsilon=1e-5, axis=(0, -1))(x, use_running_average=False)
            x = nn.BatchNorm()(x, use_running_average=not train)
            x = nn.leaky_relu(x, negative_slope=0.2)

        x = nn.Conv(1, (4, 4), (1, 1), padding=1)(x)
        return x


if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    x = jax.random.normal(rng, (1, 256, 256, 3))
    model = Discriminator()
    params = model.init(init_rng, x)
    y = model.apply(params, x, train=False)
    print(y.shape)