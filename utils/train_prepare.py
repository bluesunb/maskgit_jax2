import os, pickle
from pathlib import Path
import jax, jax.numpy as jp
import optax
from flax.training.checkpoints import restore_checkpoint
import numpy as np
from torchvision import transforms as T

from models.vqgan import VQGAN, Discriminator
from models.mvtm import VQGANTransformer
from config import VQConfig, AutoencoderConfig, TrainConfig, TransformerConfig
from utils.dataset import load_folder_data, load_stl
from utils.context import make_rngs, make_state
from utils.metrics import LPIPS


def prepare_dataset(config: TrainConfig):
    img_size = config.img_size
    batch_size = config.batch_size
    num_workers = config.num_workers

    train_transform = T.Compose([T.Resize(img_size),
                                 T.RandomCrop((img_size, img_size)),
                                 T.RandomHorizontalFlip(),
                                 T.ToTensor(),
                                 T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_transform = T.Compose([T.Resize(img_size),
                                T.CenterCrop((img_size, img_size)),
                                T.ToTensor(),
                                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def test_untransform(x: np.ndarray):
        x = x * 0.5 + 0.5
        x = np.clip(x, 0, 1)
        x = (x * 255).astype(np.uint8)
        return x

    root_dir = os.path.expanduser("~/PycharmProjects/Datasets")

    if config.dataset == "imagenet":
        path = os.path.join(root_dir, "ILSVRC2012_img_test")
        train_loader = load_folder_data(path,
                                        batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                        transform=train_transform,
                                        max_size=config.max_size)

        test_loader = load_folder_data(path,
                                       batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                       transform=test_transform,
                                       max_size=config.max_size)

    elif config.dataset == "stl":
        train_loader = load_stl(root_dir, split='train+unlabeled', shuffle=False, batch_size=batch_size,
                                num_workers=num_workers)
        test_loader = load_stl(root_dir, split='test', shuffle=False, batch_size=batch_size, num_workers=num_workers)

    return train_loader, test_loader, test_untransform


def prepare_vqgan(config: TrainConfig):
    img_size = config.img_size

    enc_config = AutoencoderConfig(out_channels=128,
                                   channel_multipliers=(1, 2, 2, 4))
    dec_config = AutoencoderConfig(out_channels=3,
                                   channel_multipliers=(1, 2, 2, 4))
    vq_config = VQConfig(codebook_size=512)

    gan = VQGAN(enc_config, dec_config, vq_config)
    discriminator = Discriminator(emb_channels=128, n_layers=3)

    v_rng, d_rng = jax.random.split(jax.random.PRNGKey(config.seed))

    vqgan_state = make_state(rngs=make_rngs(v_rng, ('dropout',), contain_params=True),
                             model=gan,
                             tx=optax.chain(optax.zero_nans(),
                                            optax.adam(config.lr, b1=config.betas[0], b2=config.betas[1])),
                             inputs=jp.empty((1, img_size, img_size, 3)),
                             train=True)

    disc_state = make_state(rngs=make_rngs(d_rng, (), contain_params=True),
                            model=discriminator,
                            tx=optax.chain(optax.zero_nans(),
                                           optax.adam(config.lr, b1=config.betas[0], b2=config.betas[1])),
                            inputs=jp.empty((1, img_size, img_size, 3)),
                            train=True)

    lpips_fn1 = load_lpips_fn(LPIPS if config.use_lpips else None, img_size, 3)
    lpips_fn2 = load_lpips_fn(LPIPS if config.use_lpips else None, img_size, 3)
    configs = {'enc': enc_config, 'dec': dec_config, 'vq': vq_config}

    return vqgan_state, disc_state, lpips_fn1, lpips_fn2, configs


def load_lpips_fn(lpips_def: LPIPS = None, img_size: int = 256, channel: int = 3):
    def identity(*args, **kwargs):
        return jp.zeros(())

    if lpips_def is None:
        return identity

    rng = jax.random.PRNGKey(0)
    lpips = LPIPS()
    params = lpips.init(rng, jp.ones((1, img_size, img_size, channel)), jp.ones((1, img_size, img_size, channel)))
    lpips = lpips.bind(params)
    return lpips.__call__


def prepare_transformer(config: TrainConfig, vq_path: Path):
    # vqgan_state = restore_checkpoint(vq_path)
    vqgan_params = pickle.load(open(vq_path, 'rb'))
    # vqgan_params = {'params': vqgan_state['params']}
    # if 'extra_variables' in vqgan_state:
    #     vqgan_params.update(vqgan_state['extra_variables'])
    vq_configs = pickle.load(open(vq_path.parent / 'vq_configs.pkl', 'rb'))
    vqgan = VQGAN(enc_config=vq_configs['enc'],
                  dec_config=vq_configs['dec'],
                  vq_config=vq_configs['vq'],
                  training=False)
    
    # vqgan = vqgan.bind(vqgan_params)

    trns_config = TransformerConfig(emb_dim=128,
                                    n_heads=8,
                                    n_layers=12,
                                    intermediate_dim=4 * 128,
                                    attn_pdrop=0.1,
                                    resid_pdrop=0.1,
                                    ff_pdrop=0.1,
                                    codebook_size=vq_configs['vq'].codebook_size,
                                    sample_temperature=vq_configs['vq'].entropy_temperature,
                                    mask_scheme="cosine")

    transformer = VQGANTransformer(trns_config, vqgan, vqgan_params)
    # transformer = VQGANTransformer(trns_config)

    scheduler = optax.cosine_onecycle_schedule(transition_steps=config.n_epochs * (config.max_size // config.batch_size),
                                               peak_value=config.lr,
                                               pct_start=0.3,
                                               div_factor=25.0,
                                               final_div_factor=5.0)
    
    rng = jax.random.PRNGKey(config.seed)
    # tx = optax.chain(optax.zero_nans(),
    #                  optax.adam(scheduler, b1=config.betas[0], b2=config.betas[1]))
    tx = optax.adam(scheduler, b1=config.betas[0], b2=config.betas[1])
    if config.grad_accum > 1:
        tx = optax.MultiSteps(tx, config.grad_accum)

    trns_state = make_state(rngs=make_rngs(rng, ('mask', 'dropout'), contain_params=True),
                            model=transformer,
                            tx=tx,
                            inputs=jp.empty((1, config.img_size, config.img_size, 3)),
                            # param_exclude=('vqgan',),
                            train=True)
    
    return trns_state, trns_config
