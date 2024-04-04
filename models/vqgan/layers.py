import jax
import jax.numpy as jp
import flax.linen as nn


default_kernel = nn.initializers.xavier_uniform()


class ResBlock(nn.Module):
    out_channels: int
    use_conv_shortcut: bool = False
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x: jp.ndarray, train: bool = True):
        out = nn.GroupNorm(num_groups=32, epsilon=1e-6)(x)
        out = nn.swish(out)
        out = nn.Conv(self.out_channels, (3, 3), padding='SAME', use_bias=False, kernel_init=default_kernel)(out)
        out = nn.GroupNorm(num_groups=32, epsilon=1e-6)(out)
        out = nn.swish(out)
        out = nn.Dropout(rate=self.dropout_rate)(out, deterministic=not train)
        out = nn.Conv(self.out_channels, (1, 1), padding='SAME', use_bias=False, kernel_init=default_kernel)(out)

        if x.shape[-1] != self.out_channels:
            if self.use_conv_shortcut:
                x = nn.Conv(self.out_channels, (3, 3), padding='SAME', use_bias=False, kernel_init=default_kernel)(x)
            x = nn.Conv(self.out_channels, (1, 1), padding='SAME', use_bias=False, kernel_init=default_kernel)(x)

        return x + out
    

class Upsample(nn.Module):
    @nn.compact
    def __call__(self, x: jp.ndarray):
        n, h, w, c = x.shape
        x = jax.image.resize(x, (n, h * 2, w * 2, c), method='nearest')
        x = nn.Conv(c, (3, 3), padding='SAME', use_bias=False)(x)
        return x
    

class Downsample(nn.Module):
    with_conv: bool = False
    @nn.compact
    def __call__(self, x: jp.ndarray):
        n, h, w, c = x.shape
        if self.with_conv:
            pad = ((0, 0), (0, 1), (0, 1), (0, 0))
            x = jp.pad(x, pad, mode='constant', constant_values=0)
            x = nn.Conv(c, (3, 3), strides=(2, 2), padding='VALID', use_bias=False)(x)
        else:
            x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID')
        
        return x


class DownsamplePool(nn.Module):
    @nn.compact
    def __call__(self, x: jp.ndarray):
        return nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID')
    

class Attention(nn.Module):
    @nn.compact
    def __call__(self, x: jp.ndarray):
        out = nn.GroupNorm(num_groups=32, epsilon=1e-6)(x)
        qkv = nn.Conv(3 * x.shape[-1], (1, 1), padding="valid", use_bias=False)(out)
        q, k, v = jp.split(qkv, 3, axis=-1)

        bs, h, w, c = q.shape
        q = q.reshape((bs, (h * w), c))
        k = k.reshape((bs, (h * w), c))
        attn_weights = jp.einsum('b q c, b k c -> b q k', q, k) / jp.sqrt(c)
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)

        v = v.reshape((bs, (h * w), c))
        out = jp.einsum('b q k, b k c -> b q c', attn_weights, v)
        out = out.reshape((bs, h, w, c))
        out = nn.Conv(c, (1, 1), padding='SAME', use_bias=False, kernel_init=default_kernel)(out)
        return out + x