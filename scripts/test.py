# import jax, jax.numpy as jp

import torch as th

def f(x):
    print('Called')
    return th.sin(x)

x = th.tensor(th.pi)
a = f(x)

l1 = (a ** 2)
l2 = th.log(a)

l1.backward()
l2.backward()

print(x.grad)