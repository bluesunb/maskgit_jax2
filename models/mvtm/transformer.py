import os
import math
import numpy as np
import jax, jax.numpy as jp
import flax.linen as nn
import optax

from models.vqgan.vqvae import VQGAN
from models.vqgan.codebook import vector_quantization
from models.mvtm.bi_transformer import BidirectionalTransformer, BidirectionalTransformer2
from config import TransformerConfig
from scripts.common import TrainState
from utils.context import load_state
from functools import partial


def get_mask_schedule(mode="cosine"):
    if mode == "linear":
        return lambda r: 1 - r
    elif mode == "cosine":
        return lambda r: jp.cos(r * jp.pi / 2)
    elif mode == "square":
        return lambda r: 1 - r ** 2
    elif mode == "cubic":
        return lambda r: 1 - r ** 3

    raise NotImplementedError(f"Mask schedule mode {mode} not implemented")


class VQGANTransformer(nn.Module):
    config: TransformerConfig
    vqgan: VQGAN
    training: bool = None

    def setup(self):
        self.sos_token = self.config.codebook_size + 1
        self.mask_token_id = self.config.codebook_size
        self.scheduler = get_mask_schedule(self.config.mask_scheme)
        self.transformer = BidirectionalTransformer(self.config)

    def encode_to_z(self, x: jp.ndarray, train: bool = True):
        x_enc = self.vqgan.encode(x)
        quantized, q_loss, result = self.vqgan.quantize(x_enc)
        indices = result["indices"]
        quantized = jax.lax.stop_gradient(quantized)
        indices = jax.lax.stop_gradient(indices)
        return quantized, indices

    @nn.compact
    def __call__(self, x: jp.ndarray, train: bool = True):
        train = nn.merge_param('training', self.training, train)
        quantized, z_indices = self.encode_to_z(x)        # z_indices: [bs, h/, w/, z_dim]
        z_indices = z_indices.reshape((x.shape[0], -1))
        sos_tokens = jp.ones((x.shape[0], 1), dtype=jp.int32) * self.sos_token

        mask_rng = self.make_rng('mask')
        n_mask_rng, mask_sample_rng = jax.random.split(mask_rng)
        n_indices = z_indices.shape[1]
        n_leave = math.floor(jax.random.uniform(n_mask_rng, ()) * n_indices)
        sample = jax.random.uniform(mask_sample_rng, z_indices.shape)
        sample = sample.argsort(1)[:, :n_leave]
        sample = jax.nn.one_hot(sample, n_indices).any(axis=1)      # where to leave the token (0: will be masked)
        mask = jp.where(sample, True, False)

        masked = mask * z_indices + (1 - mask) * self.mask_token_id
        masked = jp.concatenate([sos_tokens, masked], axis=1)   # [sos, z1, m2, ..., mn]
        # logits = BidirectionalTransformer(self.config, name='transformer')(masked, train)
        logits = self.transformer(masked, train)
        target = jp.concatenate([sos_tokens, z_indices], axis=1)    # [sos, z1, z2, ..., zn]
        return logits, target

    def sample_ids(self,
                   ids: jp.ndarray = None,
                   batch_size: int = None,
                   total_steps: int = 11,
                   schedule: str = 'cosine'):

        n_img_tokens = self.variables['params']['transformer']['bias'].shape[0] - 1
        if ids is None:
            ids = jp.ones((batch_size, n_img_tokens), dtype=jp.int32) * self.mask_token_id
        else:
            ids = jp.concatenate([ids, jp.full((ids.shape[0], n_img_tokens - ids.shape[1]), self.mask_token_id)], axis=-1)

        sos_tokens = jp.ones((ids.shape[0], 1), dtype=jp.int32)
        ids = jp.concatenate([sos_tokens, ids], axis=1)       # (bs, n_img_tokens + 1)

        n_masks = jp.sum(ids == self.mask_token_id, axis=-1)
        scheduler = get_mask_schedule(schedule)

        rng = self.make_rng('fill')

        for t in range(total_steps):
            rng, _ = jax.random.split(rng)
            ids = self.sample_forward(rng, ids, t, total_steps, n_masks, scheduler)

        return ids[..., 1:]

    def sample_forward(self,
                       args,
                       t: int,
                       total_steps: int,
                       n_masks: jp.ndarray,
                       scheduler: callable):

        rng, ids = args
        rng, _ = jax.random.split(rng)
        pred_rng, noise_rng = jax.random.split(rng)
        logits = self.transformer(ids, train=False)      # (bs, seq_len, n_tokens = codebook + 2)
        pred_ids = jax.random.categorical(rng, logits, axis=-1)     # (bs, seq_len)

        mask = (ids == self.mask_token_id)
        pred_ids = jp.where(mask, pred_ids, ids)  # fill predicted tokens to the masked positions

        r = (t + 1) / total_steps
        mask_ratio = scheduler(r)

        probs = jax.nn.softmax(logits, axis=-1)
        pred_probs = jp.take_along_axis(probs, pred_ids[..., None], axis=-1)    # (bs, seq_len, 1)
        pred_probs = jp.where(mask, pred_probs.squeeze(-1), 1e5)

        new_n_masks = jp.floor(mask_ratio * n_masks).astype(jp.int32)
        new_n_masks = jp.maximum(0, jp.minimum(new_n_masks, n_masks))[..., None]

        temperature = self.config.sample_temperature * (1 - r)
        confidence = jp.log(pred_probs) + jax.random.gumbel(noise_rng, pred_probs.shape) * temperature
        cut_off = jp.sort(confidence, axis=-1)
        cut_off = jp.take_along_axis(cut_off, new_n_masks, axis=-1)     # threshold value for masking
        new_mask = confidence < cut_off                                 # mask-out under the threshold

        ids = jp.where(new_mask, self.mask_token_id, pred_ids)
        return ids

    @partial(jax.jit, static_argnums=(2,))
    def log_images(self, x, schedule="cosine"):
        quantized, z_indices = self.encode_to_z(x)
        sampled_ids = self.sample_ids(z_indices, total_steps=11, schedule=schedule)
        x_new = self.ids_to_image(sampled_ids)

        z_half_ids = z_indices[:, :z_indices.shape[1] // 2]
        half_sampled_ids = self.sample_ids(z_half_ids, total_steps=11, schedule=schedule)
        x_sample = self.ids_to_image(half_sampled_ids)

        x_recon = self.ids_to_image(z_indices)
        return x_recon, x_sample, x_new

    def ids_to_image(self, ids):
        encodings = jax.nn.one_hot(ids, self.config.codebook_size)
        ids = vector_quantization(encodings, self.vqgan.codebook.get_codebook())
        x_recon = self.vqgan.decode(ids)
        return x_recon
    
    # def fill_mask_scan(self,
    #               inputs: jp.ndarray = None,
    #               batch_size: int = None,
    #               total_steps: int = 11,
    #               schedule: str = 'cosine'):
    #
    #     n_img_tokens = self.variables['params']['transformer']['bias'].shape[0] - 1
    #     if inputs is None:
    #         inputs = jp.ones((batch_size, n_img_tokens), dtype=jp.int32) * self.mask_token_id
    #     else:
    #         inputs = jp.concatenate(
    #             [inputs, jp.full((inputs.shape[0], n_img_tokens - inputs.shape[1]), self.mask_token_id)], axis=-1)
    #
    #     sos_tokens = jp.ones((inputs.shape[0], 1), dtype=jp.int32)
    #     inputs = jp.concatenate([sos_tokens, inputs], axis=1)  # (bs, n_img_tokens + 1)
    #
    #     n_masks = jp.sum(inputs == self.mask_token_id, axis=-1)
    #     scheduler = get_mask_schedule(schedule)
    #
    #     def mask_forward(transformer: nn.Module,
    #                      args,
    #                      t: int,
    #                      total_steps: int,
    #                      n_masks: jp.ndarray,
    #                      scheduler: callable):
    #
    #         rng, inputs = args
    #         rng, _ = jax.random.split(rng)
    #         pred_rng, noise_rng = jax.random.split(rng)
    #         logits = transformer(inputs, train=False)  # (bs, seq_len, n_tokens = codebook + 2)
    #         pred_ids = jax.random.categorical(rng, logits, axis=-1)  # (bs, seq_len)
    #
    #         mask = (inputs == self.mask_token_id)
    #         pred_ids = jp.where(mask, pred_ids, inputs)  # fill predicted tokens to the masked positions
    #
    #         r = (t + 1) / total_steps
    #         mask_ratio = scheduler(r)
    #
    #         probs = jax.nn.softmax(logits, axis=-1)
    #         pred_probs = jp.take_along_axis(probs, pred_ids[..., None], axis=-1)  # (bs, seq_len, 1)
    #         pred_probs = jp.where(mask, pred_probs.squeeze(-1), 1e5)
    #
    #         new_n_masks = jp.floor(mask_ratio * n_masks).astype(jp.int32)
    #         new_n_masks = jp.maximum(0, jp.minimum(new_n_masks, n_masks))[..., None]
    #
    #         temperature = self.config.sample_temperature * (1 - r)
    #         confidence = jp.log(pred_probs) + jax.random.gumbel(noise_rng, pred_probs.shape) * temperature
    #         cut_off = jp.sort(confidence, axis=-1)
    #         cut_off = jp.take_along_axis(cut_off, new_n_masks, axis=-1)  # threshold value for masking
    #         new_mask = confidence < cut_off  # mask-out under the threshold
    #
    #         inputs = jp.where(new_mask, self.mask_token_id, pred_ids)
    #         return (rng, inputs), 0
    #
    #     rng = self.make_rng('fill')
    #     time_steps = jp.arange(total_steps)
    #     # (rng, inputs), _ = jax.lax.scan(partial(mask_forward, total_steps=total_steps, n_masks=n_masks, scheduler=scheduler),
    #     #                                 (rng, inputs), time_steps)
    #     (rng, inputs), _ = nn.scan(partial(mask_forward, total_steps=total_steps, n_masks=n_masks, scheduler=scheduler),
    #                                variable_broadcast="params",
    #                                split_rngs={"params": False})(self.transformer, (rng, inputs), time_steps)
    #
    #     return inputs[..., 1:]

# def patch_mask(x, p):
#     mask = jax.lax.conv_general_dilated_patches(
#         x[None, None, ...],
#         filter_shape=(1, p, p),
#         window_strides=(1, p, p),
#         padding='valid'
#     )
#     d = int(mask.shape[1] ** 0.5)
#     mask = mask.reshape(d, d, *mask.shape[2:]).transpose(3, 4, 2, 0, 1)
#     return mask.reshape(mask.shape[0] * mask.shape[1], -1)


def inpainting(img: jp.ndarray,
               model_def: VQGANTransformer,
               rng: jp.ndarray,
               x_start: int = 100,
               y_start: int = 100,
               size: int = 50,
               codebook_size: int = 1024):

    mask = jp.ones_like(img, dtype=jp.int32)
    mask = mask.at[:, x_start:x_start + size, y_start:y_start + size, :].set(0)
    mask = mask[:, :, :, 0]
    img = img * mask[..., None]

    rng, id_rng = jax.random.split(rng)
    # ids = jax.random.randint(id_rng, (1, 6, 6, 256), 0, codebook_size)
    ids = jax.random.randint(rng, (1, 256), 0, codebook_size)

    p = 16
    patched_mask = jax.lax.conv_general_dilated_patches(
        mask[None, None, ...],
        filter_shape=(1, p, p),
        window_strides=(1, p, p),
        padding='valid'
    )
    d = int(patched_mask.shape[1] ** 0.5)
    patched_mask = patched_mask.reshape(d, d, *patched_mask.shape[2:]).transpose(3, 4, 2, 0, 1)
    patched_mask = patched_mask.reshape(patched_mask.shape[0] * patched_mask.shape[1], -1)
    
    ids_mask = patched_mask.min(-1)
    ids = ids * ids_mask

    sampled_ids = model_def.sample_ids(ids)
    inpainted = model_def.ids_to_image(sampled_ids)

    ids_mask = ids_mask.reshape
    # patched_mask = patched_mask.transpose(0, 1, 2, 4, 3)
    # pathced_mask = pathced_mask.transpose(1, 2, 0, 3, 4)


if __name__ == "__main__":
    from time import time
    from pprint import pp
    from config import AutoencoderConfig, VQConfig
    config = TransformerConfig(emb_dim=128,
                               n_heads=8,
                               n_layers=3,
                               intermediate_dim=4 * 128,
                               attn_pdrop=0.1,
                               resid_pdrop=0.1,
                               ff_pdrop=0.1,
                               # n_tokens=24,
                               codebook_size=240,
                               sample_temperature=4.5,
                               mask_scheme="cosine")

    vqgan = VQGAN(enc_config=AutoencoderConfig(out_channels=128),
                  dec_config=AutoencoderConfig(out_channels=3),
                  vq_config=VQConfig(codebook_size=config.codebook_size),
                  training=False)

    rng = jax.random.PRNGKey(0)
    # vq_param = vqgan.init(rng, jp.empty((2, 96, 96, 3)))
    # vqgan = vqgan.bind(vq_param)
    #
    model = VQGANTransformer(config, vqgan)
    # print(vqgan(jp.ones((2, 96, 96, 3)))[0].shape)

    rng, mask_rng = jax.random.split(rng)
    # x = jax.random.normal(rng, (2, 96, 96, 3))
    # params = model.init({'params': rng, 'mask': mask_rng, 'dropout': rng}, x, train=True)

    # def loss_fn(params, x):
    #     logits, target = model.apply(params, x, train=True, rngs={'mask': mask_rng, 'dropout': rng})
    #     return optax.softmax_cross_entropy_with_integer_labels(logits, target).mean()
    #
    # grad_fn = jax.value_and_grad(loss_fn)
    # loss, grad = grad_fn(params, x)
    # print(loss)
    #
    # pp(jax.tree_map(lambda x: jp.linalg.norm(x).item(), grad), width=180)

    import pickle
    path = "/home/bluesun/PycharmProjects/Repr_Learning/MaskGit/maskgit_jax2/models/mvtm/params.pkl"
    if os.path.exists(path):
        params = pickle.load(open(path, 'rb'))
    else:
        pickle.dump(params, open(path, 'wb'))

    rng, sample_rng = jax.random.split(rng)
    z = jax.random.randint(rng, (2, 36), 0, config.codebook_size + 1)
    # st = time()
    # out = model.apply(params, z, rngs={'mask': mask_rng, 'dropout': rng, 'fill': sample_rng},
    #                   method=model.sample_ids, batch_size=None, total_steps=11, schedule='cosine')
    # print(f"time: {time() - st:.4f}")

    # st = time()
    # out2 = model.apply(params, z, rngs={'mask': mask_rng, 'dropout': rng, 'fill': sample_rng},
    #                    method=model.fill_mask_scan, batch_size=None, total_steps=11, schedule='cosine')
    # print(f"time: {time() - st:.4f}")

    # a1 = jax.jit(partial(model.apply, method=model.sample_ids, batch_size=None, total_steps=11, schedule='cosine'))
    # a2 = jax.jit(partial(model.apply, method=model.fill_mask_scan, batch_size=None, total_steps=11, schedule='cosine'))

    # print()
    img = jax.random.normal(rng, (1, 256, 256, 3))
    out = inpainting(img, model, rng, codebook_size=config.codebook_size)