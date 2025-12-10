# inference_prompt.py

import os
import torch
import numpy as np
import scipy.io as scio
from argparse import ArgumentParser

from stablediff.params import AttrDict, params_wifi
from stablediff.wifi_model import tfdiff_WiFi
from stablediff.diffusion import SignalDiffusion, GaussianDiffusion

def load_wifi_model(model_dir, device):
    params = AttrDict(params_wifi)

    if os.path.exists(os.path.join(model_dir, "weights.pt")):
        checkpoint = torch.load(os.path.join(model_dir, "weights.pt"), map_location=device)
    else:
        checkpoint = torch.load(model_dir, map_location=device)

    model = tfdiff_WiFi(params).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, params

def build_diffusion(params):
    if params.signal_diffusion:
        return SignalDiffusion(params)
    else:
        return GaussianDiffusion(params)

def generate_from_prompt(prompt: str, model, diffusion, params, device, num_samples: int = 1):
    # cond is a list[str], length = num_samples
    cond = [prompt] * num_samples

    # diffusion.sampling now supports list[str] for WiFi
    x_hat = diffusion.sampling(model, cond, device)   # [B, N, F, 2]
    return x_hat

def save_mat(path, iq_tensor, prompt):
    """
    iq_tensor: [B, N, input_dim, 2]
    Save first sample as complex IQ in .mat
    """
    x = iq_tensor[0]                             # [N, input_dim, 2]
    x_complex = torch.view_as_complex(x)         # [N, input_dim]
    mat = {
        "iq": x_complex.cpu().numpy(),
        "prompt": np.array([prompt], dtype=object)
    }
    scio.savemat(path, mat)

def main(args):
    params = params_wifi
    model_dir = args.model_dir or params.model_dir
    out_dir = args.out_dir or params.out_dir

    if args.cond_dir is not None:
        params.cond_dir = args.cond_dir

    device = torch.device('cpu') 

    if os.path.exists(f'{model_dir}/weights.pt'):
        checkpoint = torch.load(f'{model_dir}/weights.pt', map_location=device)
    else:
        checkpoint = torch.load(model_dir)

    model = tfdiff_WiFi(AttrDict(params)).to(device)

    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.params.override(params)

    diffusion = SignalDiffusion(
        params) if params.signal_diffusion else GaussianDiffusion(params)
    
    with torch.no_grad():
        pred = diffusion.sampling(model, args.prompt, device)

    print(f"Saving to {out_dir}")

    save_mat(out_dir, pred, args.prompt)


if __name__ == '__main__':
    parser = ArgumentParser(
        description='runs inference (generation) process based on trained tfdiff model')
    parser.add_argument('--task_id', type=int, default=0,
                        help='use case of tfdiff model, 0/1/2/3 for WiFi/FMCW/MIMO/EEG respectively')
    parser.add_argument('--prompt', type=str, required=True,
                        help="prompt to give to diffusion model to evaluate"  )
    parser.add_argument('--model_dir', default=None,
                        help='directory in which to store model checkpoints')
    parser.add_argument('--out_dir', default=None,
                        help='directories from which to store genrated data file')
    parser.add_argument('--cond_dir', default=None,
                        help='directories from which to read condition files for generation')
    parser.add_argument('--device', default='cuda',
                        help='device for data generation')
    main(parser.parse_args())
