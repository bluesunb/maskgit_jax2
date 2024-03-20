import jax
import jax.numpy as jp
import flax.linen as nn

from models.vqgan.layers import ResBlock, Upsample, Downsample, DownsamplePool, Attention
from config import AutoencoderConfig

from typing import Sequence


default_kernel = nn.initializers.xavier_uniform()


class DownBlock(nn.Module):
    channels: int
    channel_multipliers: Sequence[int]
    attn_resolutions: Sequence[int]
    n_blocks: int
    dropout_rate: float = 0.0
    resample_with_conv: bool = True

    @nn.compact
    def __call__(self, hidden: jp.ndarray, train: bool = True):
        curr_size = hidden.shape[1]
        for i, mult in enumerate(self.channel_multipliers):
            c = self.channels * mult
            do_attn = curr_size in self.attn_resolutions

            for _ in range(self.n_blocks):
                hidden = ResBlock(c, dropout_rate=self.dropout_rate)(hidden, train=train)
                if do_attn:
                    hidden = Attention()(hidden)
            
            if i != len(self.channel_multipliers) - 1:
                hidden = Downsample(with_conv=self.resample_with_conv)(hidden)
                curr_size //= 2

        return hidden
    

class UpBlock(nn.Module):
    channels: int
    channel_multipliers: Sequence[int]
    attn_resolutions: Sequence[int]
    n_blocks: int
    dropout_rate: float = 0.0
    resample_with_conv: bool = True

    @nn.compact
    def __call__(self, hidden: jp.ndarray, train: bool = True):
        curr_size = hidden.shape[1]
        for i, mult in enumerate(reversed(self.channel_multipliers)):
            c = self.channels * mult
            do_attn = curr_size in self.attn_resolutions
            
            for _ in range(self.n_blocks):
                hidden = ResBlock(c, dropout_rate=self.dropout_rate)(hidden, train=train)
                if do_attn:
                    hidden = Attention()(hidden)

            if i != len(self.channel_multipliers) - 1:
                hidden = Upsample()(hidden)
                curr_size *= 2

        return hidden


class Encoder(nn.Module):
    config: AutoencoderConfig

    @nn.compact
    def __call__(self, x: jp.ndarray, train: bool = True):
        config = self.config
        hidden = nn.Conv(config.channels, (3, 3), padding='SAME', use_bias=False, kernel_init=default_kernel, name='ConvIn')(x)
        hidden = DownBlock(config.channels,
                           config.channel_multipliers,
                           config.attn_resolutions,
                           config.n_blocks,
                           config.dropout_rate,
                           config.resample_with_conv,
                           name='UpBlock')(hidden, train)
        
        c = config.channels * config.channel_multipliers[-1]
        hidden = ResBlock(c, dropout_rate=config.dropout_rate)(hidden, train)
        hidden = Attention()(hidden)
        hidden = ResBlock(c, dropout_rate=config.dropout_rate)(hidden, train)

        hidden = nn.GroupNorm(num_groups=32, epsilon=1e-6)(hidden)
        hidden = nn.swish(hidden)
        hidden = nn.Conv(self.config.out_channels, (3, 3),
                         padding='SAME', use_bias=False, kernel_init=default_kernel, name='ConvOut')(hidden)
        return hidden
    

class Decoder(nn.Module):
    config: AutoencoderConfig

    @nn.compact
    def __call__(self, x: jp.ndarray, train: bool = True):
        config = self.config
        in_channels = config.channels * config.channel_multipliers[-1]

        hidden = nn.Conv(in_channels, (3, 3), padding='SAME', use_bias=True, kernel_init=default_kernel, name='ConvIn')(x)
        hidden = ResBlock(in_channels, dropout_rate=config.dropout_rate)(hidden, train)
        hidden = Attention()(hidden)
        hidden = ResBlock(in_channels, dropout_rate=config.dropout_rate)(hidden, train)

        hidden = UpBlock(config.channels,
                         config.channel_multipliers,
                         config.attn_resolutions,
                         config.n_blocks,
                         config.dropout_rate,
                         config.resample_with_conv)(hidden, train)
        
        hidden = nn.GroupNorm(num_groups=32, epsilon=1e-6)(hidden)
        hidden = nn.swish(hidden)
        hidden = nn.Conv(config.out_channels, (3, 3),
                         padding='SAME', use_bias=False, kernel_init=default_kernel, name='ConvOut')(hidden)
        return hidden


if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (1, 96, 96, 3))
    config = AutoencoderConfig(channels=64,
                               out_channels=128,
                               channel_multipliers=[1, 1, 2, 2, 4,],
                               attn_resolutions=[24],
                               n_blocks=2,
                               dropout_rate=0.1,
                               resample_with_conv=True)
    
    encoder = Encoder(config)
    out, param = encoder.init_with_output({'params': rng, 'dropout': rng}, x, True)

    config = AutoencoderConfig(channels=64,
                               out_channels=3,
                               channel_multipliers=[1, 1, 2, 2, 4],
                               attn_resolutions=[24],
                               n_blocks=2,
                               dropout_rate=0.1,
                               resample_with_conv=True)
    
    decoder = Decoder(config)
    out, param = decoder.init_with_output({'params': rng, 'dropout': rng}, out, True)