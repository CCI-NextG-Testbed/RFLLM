# inference_prompt.py

import os
import torch
import numpy as np
import scipy.io as scio
from argparse import ArgumentParser

from stablediff.params import AttrDict, all_params
from stablediff.models import tfdiff_WiFi, tfdiff_Simple
from stablediff.diffusion import SignalDiffusion, GaussianDiffusion


def mat_to_prompt_str(v) -> str:
    """
    Robustly convert common .mat 'prompt' formats into a Python string.
    Supports:
      - object arrays like array(['text'], dtype=object)
      - bytes
      - char arrays (Nx1 or 1xN) of single characters
      - 1x1 arrays containing any of the above
    """
    if isinstance(v, str):
        return v

    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", errors="ignore")

    arr = np.asarray(v)

    # object arrays: take first element and recurse
    if arr.dtype == object:
        if arr.size == 0:
            return ""
        return mat_to_prompt_str(arr.ravel()[0])

    # MATLAB char array case (dtype like '<U1' or similar), join characters
    if arr.dtype.kind in ("U", "S"):
        # common: (1,N) or (N,1) char matrix
        if arr.ndim == 2 and (arr.shape[0] == 1 or arr.shape[1] == 1):
            return "".join(arr.ravel().tolist())
        # sometimes it's a single string in an array
        if arr.size == 1:
            return str(arr.item())
        return "".join(arr.astype(str).ravel().tolist())

    # scalar numeric? (shouldn't happen, but don't crash)
    if arr.size == 1:
        return str(arr.item())

    return str(arr)


def load_cond_from_mat(mat_path: str, prompt_key="prompt", bits_key="bits"):
    """
    Assumes mat_path is a .mat file containing:
      - prompt: string-like
      - bits: array-like (will be converted to np.uint8)
    Returns:
      prompt (str), bits (np.ndarray dtype uint8, 1D)
    """
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"MAT file not found: {mat_path}")

    mat = scio.loadmat(mat_path, verify_compressed_data_integrity=False)

    if prompt_key not in mat:
        raise KeyError(f"Missing '{prompt_key}' in {mat_path}. Keys: {sorted(mat.keys())}")
    if bits_key not in mat:
        raise KeyError(f"Missing '{bits_key}' in {mat_path}. Keys: {sorted(mat.keys())}")

    prompt = mat_to_prompt_str(mat[prompt_key])

    bits = np.asarray(mat[bits_key])
    bits = np.squeeze(bits)  # collapse MATLAB shapes (1,N), (N,1), etc.
    bits = bits.astype(np.uint8, copy=False)

    # ensure 1D vector
    if bits.ndim != 1:
        bits = bits.reshape(-1).astype(np.uint8, copy=False)

    return prompt, bits


def save_mat(path, iq_tensor, prompt, bits):
    """
    iq_tensor: [B, N, input_dim, 2]
    Save first sample as complex IQ in .mat
    Also saves transmitted bits.
    """
    x = iq_tensor[0]                     # [N, input_dim, 2]
    x_complex = torch.view_as_complex(x) # [N, input_dim]

    mat = {
        "iq": x_complex.cpu().numpy(),
        "prompt": np.array([prompt], dtype=object),
        "bits": np.asarray(bits, dtype=np.uint8),
    }
    scio.savemat(path, mat)


def main(args):
    params = all_params[args.task_id]
    model_dir = args.model_dir or params.model_dir
    out_path = args.out_dir or params.out_dir  # this is a FILE path in your current usage

    if args.cond_dir is not None:
        params.cond_dir = args.cond_dir

    # Keep cpu as you had it; change to args.device if you want GPU later
    device = torch.device("cpu")

    # Load checkpoint
    weights_path = os.path.join(model_dir, "weights.pt")
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
    else:
        checkpoint = torch.load(model_dir, map_location=device)

    # Build model
    if params.task_id == 0:
        model = tfdiff_WiFi(AttrDict(params)).to(device)
    elif params.task_id == 1:
        model = tfdiff_Simple(AttrDict(params)).to(device)
    else:
        raise ValueError(f"Unsupported task_id={params.task_id} for this script")

    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.params.override(params)

    # Build diffusion
    diffusion = SignalDiffusion(params) if params.signal_diffusion else GaussianDiffusion(params)

    # Load conditioning from .mat file (prompt + bits)
    prompt, bits = load_cond_from_mat(args.file, prompt_key="prompt", bits_key="bits")

    with torch.no_grad():
        cond = {"prompt": prompt, "bits": bits}  # bits is np.uint8 1D vector
        pred = diffusion.sampling(model, cond, device)

    print(f"Saving to {out_path}")
    save_mat(out_path, pred, prompt, bits)


if __name__ == "__main__":
    parser = ArgumentParser(description="runs inference (generation) process based on trained tfdiff model")
    parser.add_argument(
        "--task_id",
        type=int,
        default=0,
        help="use case of tfdiff model, 0/1/2/3 for WiFi/FMCW/MIMO/EEG respectively",
    )
    parser.add_argument(
        "--file",
        required=True,
        help="MAT file containing 'prompt' (string) and 'bits' (uint8 array)",
    )
    parser.add_argument(
        "--model_dir",
        default=None,
        help="directory in which to store model checkpoints",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="output .mat FILE path (or whatever your params.out_dir points to)",
    )
    parser.add_argument(
        "--cond_dir",
        default=None,
        help="directories from which to read condition files for generation",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="device for data generation (script currently uses cpu unless you change it)",
    )

    main(parser.parse_args())
