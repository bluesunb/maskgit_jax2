import jax, jax.numpy as jp

@jax.jit
def sample_n(rng, x):
    s_rng, n_rng = jax.random.split(rng)
    x = jp.broadcast_to(jp.arange(x.shape[1]), x.shape)
    x = jax.random.permutation(n_rng, x, axis=-1)
    n = jax.random.uniform(n_rng) * x.shape[1]
    x = jp.where(x < n, 1, 0)
    return x


if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (1, 10, 10))
    x = sample_n(rng, x)
    print(x)