import os, shutil
import flax, jax
import jax.numpy as jp
from scripts.common import TrainState
from flax.training import train_state, checkpoints


def save_state(state: train_state.TrainState, path: str, step: int):
    if os.path.exists(path):
        shutil.rmtree(path)
    state = flax.jax_utils.unreplicate(state)
    state = jax.device_get(state)
    state = checkpoints.save_checkpoint(path, state, step)
    return state


def load_state(path: str, state: train_state.TrainState) -> train_state.TrainState:
    state_dict = checkpoints.restore_checkpoint(path, target=state)
    return state_dict


def make_rngs(rng, names, contain_params = False):
    if not len(names):
        return jax.random.split(rng)[1]
    
    if contain_params:
        names = ('params', ) + names

    rngs = jax.random.split(rng, len(names))
    return dict(zip(names, rngs))


def make_state(rngs, model, tx, init_batch_shape):
    inputs = jp.empty(init_batch_shape)
    variables = jax.jit(model.init, static_argnames=['train'])(rngs, inputs, train=True)
    state = TrainState.create(model,
                              params=variables.pop('params'),
                              tx=tx,
                              extra_variables=variables)

    return state