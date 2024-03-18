"""Credit for https://github.com/pcuenca/lpips-j/blob/main/src/lpips_j/lpips.py"""

import h5py
import flax.linen as nn
import jax.numpy as jp

from models.third_party.flax_models import vgg
from huggingface_hub import hf_hub_download


class VGGExtractor(vgg.VGG):
    def __init__(self):
        super().__init__(
            output='activations',
            pretrained='imagenet',
            architecture='vgg16',
            include_head=False
        )
    
    def setup(self):
        weights_file = hf_hub_download(repo_id="pcuenq/lpips-jax", filename="vgg16_weights.h5")
        self.param_dict = h5py.File(weights_file, 'r')


class NetLinLayer(nn.Module):
    weights: jp.array
    kernel_size: tuple[int] = (1, 1)

    @nn.compact
    def __call__(self, x: jp.array):
        x = nn.Conv(1, self.kernel_size, kernel_init=lambda *_: self.weights, strides=None, padding=0, use_bias=False)(x)
        return x
    

class LPIPS(nn.Module):
    def setup(self):
        self.feature_names = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3']
        self.vgg = VGGExtractor()
        
        weights_file = hf_hub_download(repo_id="pcuenq/lpips-jax", filename="lpips_lin.h5")
        lin_weights = h5py.File(weights_file)
        self.lins = [NetLinLayer(weights=jp.array(lin_weights[f'lin{i}'])) for i in range(len(self.feature_names))]

    @nn.compact
    def __call__(self, inputs: jp.array, target: jp.array):
        inputs = self.vgg((inputs + 1) / 2)
        target = self.vgg((target + 1) / 2)

        # feats_input = {}
        # feats_tgt = {}
        diffs = []

        for i, feat_name in enumerate(self.feature_names):
            # feats_input[i] = normalize(inputs[feat_name])
            # feats_tgt[i] = normalize(target[feat_name])
            # diffs[i] = (feats_input[i] - feats_tgt[i]) ** 2
            feat_input = normalize(inputs[feat_name])
            feat_tgt = normalize(target[feat_name])
            diffs.append((feat_input - feat_tgt) ** 2)

        res = [spatial_average(self.lins[i](diffs[i]), keepdims=True) for i in range(len(self.feature_names))]

        val = sum(res)
        return val


def normalize(x: jp.ndarray, eps=1e-10):
    factor = jp.sqrt(jp.sum(x ** 2, axis=-1, keepdims=True))
    return x / (factor + eps)

def spatial_average(x: jp.ndarray, keepdims=True):
    """Mean over W, H"""
    return jp.mean(x, axis=(1, 2), keepdims=keepdims)


if __name__ == "__main__":
    import jax
    import jax.numpy as jp

    rng = jax.random.PRNGKey(99)
    rng_a, rng_b = jax.random.split(rng)

    model = LPIPS()
    inputs = jax.random.normal(rng_a, (1, 224, 224, 3))
    target = jax.random.normal(rng_b, (1, 224, 224, 3))

    rng, drop_rng = jax.random.split(rng)
    params = model.init({'params': rng, 'dropout': drop_rng}, inputs, target)
    print(model.apply(params, inputs, target, rngs={'dropout': drop_rng}))