import math
import jax, jax.numpy as jp
import flax.linen as nn
from chex import assert_equal_shape
from einops import rearrange
from config import TransformerConfig

default_kernel = nn.initializers.xavier_uniform()


def positional_embedding(x: jp.ndarray, emb_dim: int, maxlen: int = 512):
    pos = jp.arange(maxlen, dtype=jp.float32)[:, jp.newaxis]
    div_term = jp.exp(jp.arange(0, emb_dim, 2) * -(jp.log(10000.0) / emb_dim))
    sin_pos = jp.sin(pos * div_term)
    cos_pos = jp.cos(pos * div_term)
    pos_emb = jp.concatenate([sin_pos, cos_pos]).T
    pos_emb = jp.expand_dims(pos_emb.reshape(emb_dim, maxlen), 0)
    pos_emb = pos_emb.transpose(0, 2, 1)
    return pos_emb[:, :x.shape[1]]


class MultiHeadAttention(nn.Module):
    emb_dim: int
    n_heads: int
    attn_pdrop: float = 0.1

    def split_head(self, x: jp.ndarray):
        return rearrange(x, 'b h (n d) -> b n h d', n=self.n_heads)

    def merge_head(self, x: jp.ndarray):
        return rearrange(x, 'b n h d -> b h (n d)')

    @nn.compact
    def __call__(self, x: jp.ndarray, train: bool = True):
        qkv = nn.Dense(3 * self.emb_dim, kernel_init=default_kernel)(x)
        q, k, v = jp.split(qkv, 3, axis=-1)
        q, k, v = map(self.split_head, (q, k, v))

        attn = jp.einsum('b n q d, b n k d -> b n q k', q, k) / jp.sqrt(self.emb_dim)
        attn = jax.nn.softmax(attn, axis=-1)
        attn = nn.Dropout(rate=self.attn_pdrop)(attn, deterministic=not train)

        out = jp.einsum('b n q k, b n k d -> b n q d', attn, v)
        out = self.merge_head(out)
        out = nn.Dense(self.emb_dim, kernel_init=default_kernel)(out)
        return out


class MLP(nn.Module):
    emb_dim: int
    intermediate_dim: int
    ff_pdrop: float = 0.1

    @nn.compact
    def __call__(self, x: jp.ndarray, train: bool = True):
        x = nn.Dense(self.intermediate_dim, kernel_init=default_kernel)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.emb_dim, kernel_init=default_kernel)(x)
        x = nn.Dropout(rate=self.ff_pdrop)(x, deterministic=not train)
        return x


class TransformerEncoder(nn.Module):
    confnig: TransformerConfig

    @nn.compact
    def __call__(self, x: jp.ndarray, train: bool = True):
        config = self.confnig
        out = MultiHeadAttention(config.emb_dim, config.n_heads, config.attn_pdrop)(x, train)
        out = nn.Dropout(config.resid_pdrop)(out, deterministic=not train)
        x = nn.LayerNorm()(x + out)
        out = MLP(config.emb_dim, config.intermediate_dim, config.resid_pdrop)(x, train)
        out = nn.LayerNorm()(x + out)
        return out


class BidirectionalTransformer(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, x: jp.ndarray, train: bool = True):
        assert_equal_shape(x.shape, (None, self.config.n_tokens + 1))
        seq_len = self.config.n_tokens + 1
        n_tokens = self.config.codebook_size + 2
        emb_dim = self.config.emb_dim
        pos_embedding = self.param('pos_embedding', nn.initializers.truncated_normal(stddev=0.02), (seq_len, emb_dim))

        tok_emb = nn.Embed(n_tokens, emb_dim, name='tok_embed')(x)
        pos_emb = pos_embedding[:tok_emb.shape[1]]
        x_emb = tok_emb + pos_emb
        x_emb = nn.Dropout(0.1)(x_emb, deterministic=not train)

        for _ in range(self.config.n_layers):
            x_emb = TransformerEncoder(self.config)(x_emb, train)

        x_emb = nn.Dense(emb_dim, kernel_init=default_kernel, use_bias=False)(x_emb)
        x_emb = nn.gelu(x_emb)
        x_emb = nn.LayerNorm(epsilon=1e-12)(x_emb)

        bias = self.param('bias', nn.initializers.zeros, (seq_len, n_tokens))
        logits = jp.matmul(x_emb, self.variables['params']['tok_embed']['embedding'].T) + bias
        return logits  # (bs, seq_len, n_tokens)


if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)
    config = TransformerConfig(emb_dim=128,
                               n_heads=8,
                               n_layers=3,
                               intermediate_dim=128 * 4,
                               codebook_size=360)

    x = jax.random.randint(rng, (1, 24), 0, config.codebook_size + 1)
    model = BidirectionalTransformer(config)
    out, params = model.init_with_output({'params': rng, 'dropout': rng}, x, train=True)
    print(out.shape)
    # from easydict import EasyDict
    # x = th.randint(0, 360, (1, 24))
    # args = EasyDict(num_codebook_vectors=256, dim=128, hidden_dim=512, n_layers=3, num_image_tokens=24)
    # model = BidirectionalTransformer(args)
    # y = model(x)
    # print(y.shape)
