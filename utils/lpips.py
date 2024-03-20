import jax, jax.numpy as jp
import flax.linen as nn
from typing import Dict


class Model(nn.Module):
    config: Dict = None
