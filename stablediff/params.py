import numpy as np


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def override(self, attrs):
        if isinstance(attrs, dict):
            self.__dict__.update(**attrs)
        elif isinstance(attrs, (list, tuple, set)):
            for attr in attrs:
                self.override(attr)
        elif attrs is not None:
            raise NotImplementedError
        return self

params_simple = AttrDict(
    log_dir='./log/simple',
    model_dir='./model/simple',
    out_dir='./dataset/simple/output/prediction.mat',
    data_dir=['./dataset/simple/raw'],  # list of folders
    max_iter=None,
    inference_batch_size=1,
    robust_sampling=True,
    batch_size=1,
    learning_rate=1e-3,
    max_grad_norm=None,
    loss_alpha=0.6,
    symbol_tau=0.1,
    loss_ema_beta=0.99,
    sample_rate=2048,              # length of each signal
    input_dim=1,
    extra_dim=[1],
    embed_dim=128,
    hidden_dim=64,
    num_heads=4,
    num_block=8,
    dropout=0.0,
    mlp_ratio=4,
    learn_tfdiff=False,
    signal_diffusion=True,        # use GaussianDiffusion or set True if you prepared blur schedule
    max_text_tokens=12,
    max_step=100,
    # variance of the guassian blur applied on the spectrogram on each diffusion step [T]
    blur_schedule=((1e-5**2) * np.ones(100)).tolist(),
    # \beta_t, noise level added to the signal on each diffusion step [T]
    noise_schedule=np.linspace(1e-4, 0.003, 100).tolist(),
)
