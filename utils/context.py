import os, shutil
import flax, jax
import jax.numpy as jp
from scripts.common import TrainState
from flax.training import train_state, checkpoints

from pathlib import Path
from collections import defaultdict
from typing import Union, Sequence, Dict
from numbers import Number
from typing import Union, Sequence, Dict


def save_state(state: train_state.TrainState, path: Union[str, Path], step: int):
    if os.path.exists(path):
        shutil.rmtree(path)
    state = flax.jax_utils.unreplicate(state)
    state = jax.device_get(state)
    state = checkpoints.save_checkpoint(path, state, step)
    return state


def load_state(path: Union[str, Path], state: train_state.TrainState) -> train_state.TrainState:
    state_dict = checkpoints.restore_checkpoint(path, target=state)
    return state_dict


def make_rngs(rng: jp.ndarray,
              names: Union[Sequence[str], Dict[str, Sequence[int]]] = None,
              contain_params: bool = False):

    if names is None:
        return jax.random.split(rng)[1]

    if isinstance(names, Sequence):
        rng, *rngs = jax.random.split(rng, len(names) + 1)
        rngs = {name: r for name, r in zip(names, rngs)}
        if contain_params:
            rngs['params'] = rng
        return rngs

    rngs = jax.random.split(rng, len(names))
    return {name: make_rngs(r, names[name], contain_params) for name, r in zip(names, rngs)}


def make_state(rngs, model, tx, inputs, param_exclude=None, **kwargs):
    variables = jax.jit(model.init, static_argnames=['train'])(rngs, inputs, **kwargs)
    # variables = model.init(rngs, inputs, **kwargs)
    if param_exclude is not None:
        for key in param_exclude:
            if key in variables['params']:
                variables['params'].pop(key)

    state = TrainState.create(model,
                              params=variables.pop('params'),
                              tx=tx,
                              extra_variables=variables)

    return state


def unreplicate_dict(d):
    return jax.tree_map(lambda x: jax.device_get(x[0]), d)


class Logger:
    def __init__(self):
        self.log_dict = defaultdict(list)
        self.steps = defaultdict(list)

    def log(self, info: dict, step: int):
        for k, v in info.items():
            if hasattr(v, 'ndim') and v.ndim == 0:
                self.log_dict[k].append(v.item())
                self.steps[k].append(step)
            elif isinstance(v, Number):
                self.log_dict[k].append(v)
                self.steps[k].append(step)

    def finish(self):
        pass