import jax, jax.numpy as jp
import optax

from scripts.common import TrainState
from config import LossWeights
from utils.context import make_rngs


def vanila_d_loss(logits_real, logits_fake = None):
    if logits_real is None:
        loss_fake = jp.mean(jax.nn.softplus(-logits_fake)) * 2
        loss_real = 0
    else:
        loss_fake = jp.mean(jax.nn.softplus(logits_fake))
        loss_real = jp.mean(jax.nn.softplus(-logits_real))

    return loss_real, loss_fake


def hinge_d_loss(logits_fake, logits_real = None):
    loss_fake = jp.mean(jax.nn.relu(1.0 + logits_fake))
    if logits_real is None:
        loss_real = 0
    else:
        loss_real = jp.mean(jax.nn.relu(1.0 - logits_real))

    return loss_real, loss_fake


def train_step(vqgan_state: TrainState,
               disc_state: TrainState,
               batch: jp.ndarray,
               rng: jp.ndarray,
               config: LossWeights,
               lpips_fn: callable,
               pmap_axis: str = None):

    rng_names = {'vqgan': ('dropout', ), 'disc': ()}
    rngs_vq = make_rngs(rng, rng_names['vqgan'])
    rngs_disc = make_rngs(rng, rng_names['disc'])

    def loss_fn_nll(params):
        x_recon, vq_loss, vq_result = vqgan_state(batch, train=True, params=params, rngs=rngs_vq)
        l1_loss = jp.abs(batch - x_recon).mean()
        l2_loss = optax.l2_loss(batch, x_recon).mean()
        percept_loss = lpips_fn(batch, x_recon).mean()

        nll_loss = (l1_loss * config.log_laplace_weight +
                    l2_loss * config.log_gaussian_weight +
                    percept_loss * config.percept_weight)

        nll_loss += vq_loss * config.codebook_weight
        result = {'vq': vq_result,
                  'percept_loss': percept_loss,
                  'l1_loss': l1_loss,
                  'l2_loss': l2_loss}

        return nll_loss, result

    def loss_fn_gen(params):
        x_recon, vq_loss, result = vqgan_state(batch, train=True, params=params, rngs=rngs_vq)
        logits_fake = disc_state(x_recon, train=False, rngs=rngs_disc)
        g_loss = jp.mean(jax.nn.softplus(-logits_fake))      # enlarge logits_fake -> pred to be real
        return g_loss
        
    def loss_fn_disc(params):
        x_recon, vq_loss, vq_result = vqgan_state(batch, train=False, rngs=rngs_vq)
        x_recon = jax.lax.stop_gradient(x_recon)

        logits_fake, updates = disc_state(x_recon, train=True, rngs=rngs_disc,
                                          params=params, mutable=['batch_stats'])
        logits_real, updates = disc_state(batch, train=True, rngs=rngs_disc,
                                          params=params, extra_variables=updates, mutable=['batch_stats'])
        
        def gen_loss_fn(logits_real, logits_fake):
            # loss for generator -> lowering it will increase the logits_fake -> good for gen
            loss_real = jp.mean(jax.nn.softplus(logits_real))
            loss_fake = jp.mean(jax.nn.softplus(-logits_fake))
            return loss_real, loss_fake
        
        def disc_loss_fn(logits_real, logits_fake):
            # loss for discriminator -> lowering it will increase the logits_real -> good for disc
            loss_real = jp.mean(jax.nn.softplus(-logits_real))
            loss_fake = jp.mean(jax.nn.softplus(logits_fake))
            return loss_real, loss_fake
        
        do_flip = disc_state.step < config.disc_flip_end
        is_gen_step = jp.logical_and(jp.mod(disc_state.step, 3) == 0, do_flip)  # 66%: disc step, 33%: gen step
        loss_real, loss_fake = jax.lax.cond(is_gen_step, gen_loss_fn, disc_loss_fn, logits_real, logits_fake)

        do_disc_update = disc_state.step > config.disc_d_start
        disc_loss = (loss_real + loss_fake) * 0.5 * do_disc_update
        info = {'loss_real': loss_real, 'loss_fake': loss_fake, 'disc_update': do_disc_update, 'is_gen_step': is_gen_step}
        
        return disc_loss, (updates, info)
    
    (nll_loss, result), nll_grads = jax.value_and_grad(loss_fn_nll, has_aux=True)(vqgan_state.params)
    g_loss, g_grads = jax.value_and_grad(loss_fn_gen)(vqgan_state.params)
    (d_loss, (updates, d_result)), d_grads = jax.value_and_grad(loss_fn_disc, has_aux=True)(disc_state.params)

    result['gan'] = d_result

    last_layer_nll = jp.linalg.norm(nll_grads['decoder']['ConvOut']['kernel'])
    last_layer_gen = jp.linalg.norm(g_grads['decoder']['ConvOut']['kernel'])

    do_g_grad = jp.where(vqgan_state.step > config.disc_g_start, config.adversarial_weight, 0.0)
    g_grad_weight = last_layer_nll / (last_layer_gen + 1e-4) * do_g_grad
    g_grad_weight = jp.clip(g_grad_weight, 0.0, 1e4)

    g_grads = jax.tree_map(lambda x: x * g_grad_weight, g_grads)

    if pmap_axis is not None:
        nll_grads = jax.lax.pmean(nll_grads, axis_name=pmap_axis)
        g_grads = jax.lax.pmean(g_grads, axis_name=pmap_axis)
        d_grads = jax.lax.pmean(d_grads, axis_name=pmap_axis)

        nll_loss = jax.lax.pmean(nll_loss, axis_name=pmap_axis)
        g_loss = jax.lax.pmean(g_loss, axis_name=pmap_axis)
        d_loss = jax.lax.pmean(d_loss, axis_name=pmap_axis)

        result = jax.lax.pmean(result, axis_name=pmap_axis)

    g_grads = jax.tree_map(lambda x, y: x + y, g_grads, nll_grads)
    vqgan_state = vqgan_state.apply_gradients(g_grads)
    disc_state = disc_state.apply_gradients(d_grads, extra_variables=updates)

    result.update({'nll_loss': nll_loss, 'g_loss': g_loss, 'd_loss': d_loss,
                   'last_layer_nll': last_layer_nll, 'last_layer_gen': last_layer_gen,
                   'g_grad_weight': g_grad_weight,
                   'do_g_grad': do_g_grad})
    
    return (vqgan_state, disc_state), result


def reconstruct_image(vqgan_state: TrainState,
                      batch: jp.ndarray, 
                      rng: jp.ndarray,
                      lpips_fn: callable):
    
    rng_names = {'vqgan': ('dropout', ), 'disc': ()}
    rngs_vq = make_rngs(rng, rng_names['vqgan'])
    rngs_disc = make_rngs(rng, rng_names['disc'])

    x_recon, _, _ = vqgan_state(batch, train=False, rngs=rngs_vq)
    recon_loss = optax.l2_loss(batch, x_recon).mean()
    percept_loss = lpips_fn(batch, x_recon).mean()
    info = {'recon/recon_loss': recon_loss, 'recon/percept_loss': percept_loss}
    return x_recon, info
