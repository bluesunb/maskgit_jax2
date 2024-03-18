"""Credit for https://github.com/matthias-wright/flaxmodels/blob/main/flaxmodels/vgg/vgg.py"""

import jax
import jax.numpy as jp
import flax.linen as nn

import h5py
import warnings
import models.third_party.utils as utils

from functools import partial
from typing import Any

URLS = {'vgg16': 'https://www.dropbox.com/s/ew3vhtlg5kks8mz/vgg16_weights.h5?dl=1',
        'vgg19': 'https://www.dropbox.com/s/1sn02fnkj579u1w/vgg19_weights.h5?dl=1'}


class VGG(nn.Module):
    output: str = 'softmax'
    pretrained: str = 'imagenet'
    normalize: bool = True
    architecture: str = 'vgg16'
    include_head: bool = True
    num_classes: int = 1000
    kernel_init: Any = nn.initializers.lecun_normal()
    bias_init: Any = nn.initializers.zeros
    ckpt_dir: str = None
    dtype: str = 'float32'

    def setup(self):
        self.param_dict = None
        if self.pretrained == "imagenet":
            ckpt_file = utils.download(self.ckpt_dir, URLS[self.architecture])
            self.param_dict = h5py.File(ckpt_file, 'r')

    @nn.compact
    def __call__(self, x: jp.ndarray, train: bool = True):
        # x: (B, H, W, C)
        if self.output not in ['softmax', 'log_softmax', 'logits', 'activations']:
            raise ValueError(
                f"Invalid output type: {self.output}. Must be one of ['softmax', 'log_softmax', 'logits', 'activations']")

        if self.pretrained is not None and self.pretrained != "imagenet":
            warnings.warn(f"Pretrained weights not available for {self.pretrained}. Using random initialization.")

        if self.include_head and (x.shape[1] != 224 or x.shape[2] != 224):
            raise ValueError(f"If include_head is True, input shape must be (B, 224, 224, C). Got {x.shape}.")

        if self.normalize:
            mean = jp.expand_dims(jp.array([0.485, 0.456, 0.406]), axis=(0, 1, 2)).astype(self.dtype)
            std = jp.expand_dims(jp.array([0.229, 0.224, 0.225]), axis=(0, 1, 2)).astype(self.dtype)
            x = (x - mean) / std

        if self.pretrained == "imagenet":
            if self.num_classes != 1000:
                warnings.warn(f"Pretrained weights are for 1000 classes but {self.num_classes}. "
                              f"This will be overwritten with 1000.")
            num_classes = 1000
        else:
            num_classes = self.num_classes

        activations = {}
        mid_n_layers = 3 if self.architecture == 'vgg16' else 4

        x = self._conv_block(x, 64, 2, 1, activations=activations, dtype=self.dtype)
        x = nn.max_pool(x, (2, 2), (2, 2))
        x = self._conv_block(x, 128, 2, 2, activations=activations, dtype=self.dtype)
        x = nn.max_pool(x, (2, 2), (2, 2))
        x = self._conv_block(x, 256, mid_n_layers, 3, activations=activations, dtype=self.dtype)
        x = nn.max_pool(x, (2, 2), (2, 2))
        x = self._conv_block(x, 512, mid_n_layers, 4, activations=activations, dtype=self.dtype)
        x = nn.max_pool(x, (2, 2), (2, 2))
        x = self._conv_block(x, 512, mid_n_layers, 5, activations=activations, dtype=self.dtype)
        x = nn.max_pool(x, (2, 2), (2, 2))

        if self.include_head:
            x = jp.transpose(x, (0, 3, 1, 2))  # (B, C, H, W)
            x = jp.reshape(x, (x.shape[0], -1))
            x = self._fc_block(x, 4096, 6, activations=activations, relu=True, train=train, dtype=self.dtype)
            x = self._fc_block(x, 4096, 7, activations=activations, relu=True, train=train, dtype=self.dtype)
            x = self._fc_block(x, num_classes, 8, activations=activations, relu=False, train=train, dtype=self.dtype)

        if self.output == "activations":
            return activations

        if self.output == "softmax" and self.include_head:
            x = nn.softmax(x)

        if self.output == "log_softmax" and self.include_head:
            x = nn.log_softmax(x)

        return x

    def _conv_block(self,
                    x: jp.ndarray,
                    features: int,
                    n_layers: int,
                    block_id: str,
                    activations: dict,
                    dtype: str = 'float32'):

        for i in range(n_layers):
            layer_name = f'conv{block_id}_{i + 1}'
            w = self.kernel_init if self.param_dict is None else lambda *_: jp.array(
                self.param_dict[layer_name]['weight'])
            b = self.bias_init if self.param_dict is None else lambda *_: jp.array(self.param_dict[layer_name]['bias'])
            x = nn.Conv(features, (3, 3), kernel_init=w, bias_init=b, padding='same', name=layer_name, dtype=dtype)(x)
            x = nn.relu(x)

            activations[layer_name] = x
            activations[f'relu{block_id}_{i + 1}'] = x

        return x

    def _fc_block(self,
                  x: jp.ndarray,
                  features: int,
                  block_id: str,
                  activations: dict,
                  relu: bool = False,
                  train: bool = True,
                  dtype: str = 'float32'):

        layer_name = f'fc{block_id}'
        w = self.kernel_init if self.param_dict is None else lambda *_: jp.array(self.param_dict[layer_name]['weight'])
        b = self.bias_init if self.param_dict is None else lambda *_: jp.array(self.param_dict[layer_name]['bias'])
        x = nn.Dense(features, kernel_init=w, bias_init=b, name=layer_name, dtype=dtype)(x)
        if relu:
            x = nn.relu(x)
            activations[f'relu{block_id}'] = x
        if train:
            x = nn.Dropout(0.5)(x, deterministic=True)

        activations[layer_name] = x
        return x


def vgg16(output: str = 'softmax',
          pretrained: str = 'imagenet',
          normalize: bool = True,
          include_head: bool = True,
          num_classes: int = 1000,
          kernel_init: Any = nn.initializers.lecun_normal(),
          bias_init: Any = nn.initializers.zeros,
          ckpt_dir: str = None,
          dtype: str = 'float32'):
    return VGG(output=output,
               pretrained=pretrained,
               normalize=normalize,
               architecture='vgg16',
               include_head=include_head,
               num_classes=num_classes,
               kernel_init=kernel_init,
               bias_init=bias_init,
               ckpt_dir=ckpt_dir,
               dtype=dtype)


def vgg19(output: str = 'softmax',
          pretrained: str = 'imagenet',
          normalize: bool = True,
          include_head: bool = True,
          num_classes: int = 1000,
          kernel_init: Any = nn.initializers.lecun_normal(),
          bias_init: Any = nn.initializers.zeros,
          ckpt_dir: str = None,
          dtype: str = 'float32'):
    return VGG(output=output,
               pretrained=pretrained,
               normalize=normalize,
               architecture='vgg19',
               include_head=include_head,
               num_classes=num_classes,
               kernel_init=kernel_init,
               bias_init=bias_init,
               ckpt_dir=ckpt_dir,
               dtype=dtype)