import jax, jax.numpy as jp
import flax.linen as nn


# class ResModule(nn.Module):
#     @nn.compact
#     def __call__(self, x, i):
#         c = 2 ** i
#         h = nn.Conv(c, (3, 3), padding='SAME')(x)
#         h = nn.relu(h)
#         x = nn.Conv(c, (3, 3), padding='SAME')(x)
#         return x + h, i + 1
#
# class ResMLP(nn.Module):
#     n_layers: int = 4
#
#     @nn.compact
#     def __call__(self, x):
#         scan_mlp = nn.scan(ResModule,
#                            variable_axes={'params', 0},
#                            variable_broadcast=False,
#                            split_rngs={'params': True},
#                            length=self.n_layers)
#
#         x, i = scan_mlp()(x, 0)
#         return x


class MLP(nn.Module):
    @nn.compact
    def __call__(self, x, c):
        d = x.shape[-1]
        h = nn.Dense(features= 2 * d)(x)
        y = nn.Dense(features=2 * d)(h)
        return y, c + 1

class ScanMLP(nn.Module):
    @nn.compact
    def __call__(self,xs, c):
        scan = nn.scan(
            MLP,
            variable_carry="batch_stats",
            variable_broadcast="params",
            split_rngs={"params": False},
        )

        is_initializing = "batch_stats" not in self.variables

        return scan(name="MLP")(xs, c)


if __name__ == "__main__":
    import chex

    with chex.fake_jit():
        rng = jax.random.PRNGKey(0)
        xs = jp.stack([jp.ones((10, 1)) + i for i in range(5)])
        scan_mlp = ScanMLP()
        out, params = scan_mlp.init_with_output(rng, xs, jp.arange(5).reshape(-1, 1))