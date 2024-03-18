import jax
import flax
import flax.linen as nn
from flax import struct
import optax
from functools import partial
from typing import Any, Tuple, Optional, Callable, Union

nonpytree_fields = partial(struct.field, pytree_node=False)
Params = flax.core.FrozenDict[str, Any]


class TrainState(struct.PyTreeNode):
    step: int
    apply_fn: Callable[..., Any] = nonpytree_fields()
    model_def: Any = nonpytree_fields()
    params: Params
    extra_variables: Optional[Params]
    tx: Optional[optax.GradientTransformation] = nonpytree_fields()
    opt_state: Optional[optax.OptState] = None

    @classmethod
    def create(cls,
               model_def: nn.Module,
               params: flax.core.FrozenDict[str, Any],
               tx: Optional[optax.GradientTransformation] = None,
               extra_variables: Optional[Params] = None,
               **kwargs) -> 'TrainState':

        opt_state = None
        if tx is not None:
            opt_state = tx.init(params)

        if extra_variables is None:
            extra_variables = flax.core.FrozenDict()

        return cls(step=1,
                   apply_fn=model_def.apply,
                   model_def=model_def,
                   params=params,
                   extra_variables=extra_variables,
                   tx=tx,
                   opt_state=opt_state,
                   **kwargs)

    def __call__(self,
                 *args,
                 params=None,
                 extra_variables: dict = None,
                 method: Union[str, Callable, None] = None,
                 **kwargs):

        if params is None:
            params = self.params

        if extra_variables is None:
            extra_variables = self.extra_variables
        variables = {'params': params, **extra_variables}

        if isinstance(method, str):
            method = getattr(self.model_def, method)

        return self.apply_fn(variables, *args, method=method, **kwargs)

    def apply_gradients(self, grads, **kwargs) -> "TrainState":
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs
        )

    def apply_loss_fn(self, loss_fn, pmap_axis=None, has_aux=False) -> Tuple['TrainState', Any]:
        if has_aux:
            grads, info = jax.grad(loss_fn, has_aux=True)(self.params)
            if pmap_axis is not None:
                grads = jax.lax.pmean(grads, axis_name=pmap_axis)
                info = jax.lax.pmean(info, axis_name=pmap_axis)
            return self.apply_gradients(grads=grads), info

        else:
            grads = jax.grad(loss_fn)(self.params)
            if pmap_axis is not None:
                grads = jax.lax.pmean(grads, axis_name=pmap_axis)
            return self.apply_gradients(grads=grads)
