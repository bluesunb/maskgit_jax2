import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as plt_backend_agg
import seaborn as sns

import jax
import jax.numpy as jp
import wandb
from itertools import product
from typing import Dict, Union


def plot_to_img(run: wandb.run, tag: str, plot: plt.Figure, step: int):
    # plot.canvas.draw()
    # img = np.frombuffer(plot.canvas.tostring_rgb(), dtype=np.uint8)
    # img = img.reshape(plot.canvas.get_width_height()[::-1] + (3,))
    # img = th.from_numpy(img.setflags(write=True)).permute(2, 0, 1)
    # writer.add_image(tag, img, step)
    # plt.close(plot)
    canvas = plt_backend_agg.FigureCanvasAgg(plot)
    canvas.draw()
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    w, h = plot.canvas.get_width_height()
    img = img.reshape([h, w, -1])
    # img = th.from_numpy(img).permute(2, 0, 1)
    plt.close(plot)
    run.log({tag: wandb.Image(img)}, step=step)
    return img


def array_to_img(run: wandb.run, tag: str, arr: jp.ndarray, step: int):
    fig, ax = plt.subplots()
    plt.imshow(jax.device_get(arr))
    return plot_to_img(run, tag, fig, step)


def array_to_heatmap(run: wandb.run, tag: str, arr: jp.ndarray, step: int):
    fig, ax = plt.subplots()
    sns.heatmap(jax.device_get(arr), ax=ax)
    return plot_to_img(run, tag, fig, step)