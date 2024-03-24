import jax
import jax.numpy as jp
import flax.linen as nn
from models.vqgan.autoencoder import Encoder, Decoder
from models.vqgan.codebook import VectorQuantizer, GumbelVQ
from config import AutoencoderConfig, VQConfig


# def calc_lambda(nll_loss_fn, g_loss_fn, **kwargs):
#     nll_grad_fn = jax.grad(nll_loss_fn)
#     g_grad_fn = jax.grad(g_loss_fn)
#     nll_grad = nll_grad_fn(**kwargs)
#     g_grad = g_grad_fn(**kwargs)
#
#     last_layer_name = 'params.decoder.ConvOut.kernel'
#     nll_grad_norm = jp.

def calculate_lambda(nll_loss, g_loss, last_layer_weight):
    # Compute gradients of nll_loss and g_loss with respect to last_layer_weight
    nll_grads = grad(lambda w: nll_loss(w).sum())(last_layer_weight)
    g_grads = grad(lambda w: g_loss(w).sum())(last_layer_weight)

    # Compute the norm of gradients
    nll_grads_norm = jp.linalg.norm(nll_grads)
    g_grads_norm = jp.linalg.norm(g_grads)

    # Compute λ
    lambda_val = nll_grads_norm / (g_grads_norm + 1e-4)

    # Clamp λ between 0 and 1e4
    lambda_val = jax.clamp(lambda_val, 0, 1e4)

    # Detach λ from the computation graph
    lambda_val = jp.detach(lambda_val)

    # Scale λ by 0.8
    lambda_val *= 0.8

    return lambda_val


class VQGAN(nn.Module):
    enc_config: AutoencoderConfig
    dec_config: AutoencoderConfig
    vq_config: VQConfig
    train: bool = True

    def setup(self):
        self.n_tokens = self.vq_config.codebook_size
        self.emb_dim = self.enc_config.out_channels
        self.encoder = Encoder(self.enc_config)
        self.decoder = Decoder(self.dec_config)
        self.vq = VectorQuantizer(self.vq_config)

        self.conv = nn.Conv(self.emb_dim, kernel_size=(1, 1))
        self.post_conv = nn.Conv(self.emb_dim, kernel_size=(1, 1))

    def encode(self, x: jp.ndarray):
        x_enc = self.encoder(x, self.train)
        x_enc = self.conv(x_enc)
        # quantized, result = self.vq(x_enc, train)
        # vq_loss = result.pop('vq_loss', 0.0)
        # return quantized, vq_loss, result
        return x_enc

    def decode(self, x: jp.ndarray,):
        x = self.post_conv(x)
        return self.decoder(x, self.train)
    
    def quantize(self, x: jp.ndarray):
        quantied, result = self.vq(x, self.train)
        vq_loss = result.pop('vq_loss', 0.0)
        return quantied, vq_loss, result

    def __call__(self, x: jp.ndarray):
        # quantized, loss, result = self.encode(x, train)
        x_enc = self.encode(x)
        quantized, loss, result = self.quantize(x_enc)
        x_rec = self.decode(quantized)
        return x_rec, loss, result



if __name__ == "__main__":
    import optax
    from pprint import pp

    enc_config = AutoencoderConfig(channels=64,
                                   out_channels=128,
                                   channel_multipliers=[1, 1, 2, 2, 4,],
                                   attn_resolutions=[24],
                                   n_blocks=2,
                                   dropout_rate=0.1,
                                   resample_with_conv=True)
    
    dec_config = AutoencoderConfig(channels=64,
                                   out_channels=3,
                                   channel_multipliers=[1, 1, 2, 2, 4,],
                                   attn_resolutions=[24],
                                   n_blocks=2,
                                   dropout_rate=0.1,
                                   resample_with_conv=True)

    vq_config = VQConfig(codebook_size=256,
                         commit_loss_weight=1.0,
                         entropy_loss_weight=0.25,
                         entropy_temperature=0.5)
    
    model = VQGAN(enc_config, dec_config, vq_config, train=True)

    rng = jax.random.PRNGKey(0)
    x = jp.ones((1, 256, 256, 3))
    params = model.init({'params': rng, 'dropout': rng}, x)
    def loss_fn(params, x):
        x_rec, loss, result = model.apply(params, x, rngs={'dropout': rng})
        return optax.l2_loss(x, x_rec).mean() + loss

    grad_fn = jax.grad(loss_fn)
    grad = grad_fn(params, x)
    pp(jax.tree_map(jp.shape, grad))
