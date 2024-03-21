import jax
import jax.numpy as jp
import flax.linen as nn
import optax

from functools import partial
from config import VQConfig
from typing import Any


def uniform_init(lower, upper):
    def init(key, shape, dtype=jp.float32):
        return jax.random.uniform(key, shape, dtype, lower, upper)
    return init


def cdist(x: jp.ndarray, y: jp.ndarray) -> jp.ndarray:
    return jp.linalg.norm(jp.expand_dims(x, axis=1) - jp.expand_dims(y, axis=0), axis=-1)


def vector_indexization(inputs: jp.ndarray, codebook: jp.ndarray) -> jp.ndarray:
    distances = cdist(inputs.reshape((-1, inputs.shape[-1])), codebook)
    indices = jp.argmin(distances, axis=-1)
    indices = indices.reshape(inputs.shape[:-1])     # (bs, seq_len)
    return indices, distances


def vector_quantization(encodings: jp.ndarray, codebook: jp.ndarray) -> jp.ndarray:
    return jp.dot(encodings, codebook)


def vq_embedding(ids: jp.ndarray, codebook: jp.ndarray) -> jp.ndarray:
    return jp.take(codebook, ids, axis=0)


def calc_codebook_loss(x: jp.ndarray, quantized: jp.ndarray) -> jp.ndarray:
    return optax.l2_loss(jax.lax.stop_gradient(x), quantized).mean()


def calc_commit_loss(x: jp.ndarray, quantized: jp.ndarray) -> jp.ndarray:
    return optax.l2_loss(x, jax.lax.stop_gradient(quantized)).mean()


def calc_entropy_loss(affinity: jp.ndarray, temperature=1.0, eps=1e-5):
    """
    Args:
        affinity: this is the output of the model, the logits
        loss_type: softmax or sigmoid
        temperature: temperature for the softmax
    """
    flat_affinity = affinity.reshape((-1, affinity.shape[-1]))
    flat_affinity /= temperature

    probs = jax.nn.softmax(flat_affinity, axis=-1)
    log_probs = jax.nn.log_softmax(flat_affinity + eps, axis=-1)

    target_probs = probs

    avg_probs = jp.mean(target_probs, axis=0)
    avg_entropy = -jp.sum(avg_probs * jp.log(avg_probs + eps))
    sample_entropy = -jp.mean(jp.sum(target_probs * log_probs, axis=-1))
    loss = sample_entropy - avg_entropy
    return loss


def get_perplexity(x: jp.ndarray, axis_num="batch"):
    x = x.reshape((-1, x.shape[-1]))
    x_probs = jp.mean(x, axis=0).astype(jp.float32)
    device_probs = jax.lax.pmean(x_probs, axis_name=axis_num)
    device_perplexity = jax.exp(-jp.sum(device_probs * jp.log(device_probs + 1e-5)))
    perplexity = jp.exp(-jp.sum(x_probs * jp.log(x_probs + 1e-5)))
    return device_perplexity, perplexity


class VectorQuantizer(nn.Module):
    config: VQConfig
    dtype: Any = jp.float32

    @nn.compact
    def __call__(self, x: jp.ndarray, train: bool = True):
        # x: (bs, h, w, emb_channels)
        n_tokens = self.config.codebook_size
        codebook = self.param("codebook", 
                              nn.initializers.variance_scaling(scale=1.0, mode="fan_in", distribution="uniform"), 
                              (n_tokens, x.shape[-1])).astype(self.dtype)
        
        indices, distances = vector_indexization(x, codebook)  # (bs, seq_len)
        encodings = jax.nn.one_hot(indices, n_tokens, dtype=self.dtype)   # (bs, seq_len, n_tokens)
        quantized = vector_quantization(encodings, codebook)    # (bs, seq_len, emb_dim)
        result = {}

        if train:
            commit_loss = calc_commit_loss(x, quantized) * self.config.commit_loss_weight
            codebook_loss = calc_codebook_loss(x, quantized)
            entropy_loss = 0.0

            if self.config.entropy_loss_weight > 0:
                entropy_loss = calc_entropy_loss(-distances, temperature=self.config.entropy_temperature)
                entropy_loss = entropy_loss * self.config.entropy_loss_weight
            
            loss = codebook_loss + commit_loss + entropy_loss
            quantized = x + jax.lax.stop_gradient(quantized - x)
            result.update({'vq_loss': loss, 'commit_loss': commit_loss, 'codebook_loss': codebook_loss, 'entropy_loss': entropy_loss})

        result.update({"encodings": encodings, "indices": indices})
        return quantized, result
    
    def get_codebook(self):
        return self.variables['params']['codebook'].astype(self.dtype)
    

class GumbelVQ(nn.Module):
    config: VQConfig
    dtype: Any = jp.float32

    @nn.compact
    def __call__(self, x: jp.ndarray, tau=1.0, train: bool = True):
        n_tokens = self.config.codebook_size
        codebook = self.param("codebook", 
                              nn.initializers.variance_scaling(scale=1.0, mode="fan_in", distribution="uniform"), 
                              (n_tokens, x.shape[-1])).astype(self.dtype)
        
        indices, distances = vector_indexization(x, codebook)
        reuslt = {}
        
        if train:
            noise = jax.random.gumbel(self.make_rng("gumbel"), distances.shape, dtype=self.dtype)
            encodings = jax.nn.softmax((-distances + noise) / tau, axis=-1)
        else:
            encodings = jax.nn.one_hot(indices, n_tokens, dtype=self.dtype)
        
        quantized = vector_quantization(encodings, codebook)
        reuslt.update({"vq_loss": 0.0, "encodings": encodings, "indices": indices})
        return quantized, reuslt
    
    def get_codebook(self):
        return self.variables['params']['codebook'].astype(self.dtype)
