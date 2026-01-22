import numpy as np
import os
import random
import torch
import torch.nn.functional as F
import scipy.io as scio
from stablediff.params import AttrDict
from glob import glob
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# data_key='csi_data',
# gesture_key='gesture',
# location_key='location',
# orient_key='orient',
# room_key='room',
# rx_key='rx',
# user_key='user',

def _nested_map(struct, map_fn):
    if isinstance(struct, tuple):
        return tuple(_nested_map(x, map_fn) for x in struct)
    if isinstance(struct, list):
        return [_nested_map(x, map_fn) for x in struct]
    if isinstance(struct, dict):
        return {k: _nested_map(v, map_fn) for k, v in struct.items()}
    return map_fn(struct)


class WiFiDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        super().__init__()
        self.filenames = []
        for path in paths:
            self.filenames += glob(f'{path}/**/*.mat', recursive=True)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        cur_filename = self.filenames[idx]
        cur_sample = scio.loadmat(cur_filename,verify_compressed_data_integrity=False)
        # Ensure required keys exist
        if 'data' not in cur_sample:
            raise KeyError(f"Missing required key 'data' in file {cur_filename}")
        if 'label' not in cur_sample:
            raise KeyError(f"Missing required key 'label' in file {cur_filename}")
        if 'bits' not in cur_sample:
            raise KeyError(f"Missing required key 'bits' in file {cur_filename}")

        cur_data = torch.from_numpy(cur_sample['data']).to(torch.complex64)
        bits = cur_sample['bits']

        return {
            'data': cur_data,
            'label': cur_sample['label'],    # textual prompt or label
            'bits': bits                    # required: numpy array or similar
        }

class SimpleSignalDataset(torch.utils.data.Dataset):
    """
    Expects each .mat file to contain:
      - 'data': complex IQ (complex64/complex128 OR real/imag last-dim=2)
      - 'bits': 0/1 bits (int/float), length ~= N
      - 'label': text prompt (string / char array)
    """
    def __init__(self, paths):
        super().__init__()
        if isinstance(paths, (str, os.PathLike)):
            paths = [str(paths)]

        self.filenames = []
        for p in paths:
            self.filenames += glob(f"{p}/**/*.mat", recursive=True)

        if len(self.filenames) == 0:
            raise RuntimeError(f"No .mat files found under: {paths}")

    def __len__(self):
        return len(self.filenames)

    @staticmethod
    def _mat_to_str(x) -> str:
        # Handles MATLAB char arrays and scalar arrays
        if isinstance(x, str):
            return x
        if isinstance(x, bytes):
            return x.decode("utf-8", errors="ignore")
        if isinstance(x, np.ndarray):
            # scalar string
            if x.shape == () and x.dtype.kind in ("U", "S"):
                return str(x.item())
            # MATLAB char array e.g. shape (1,155) of single chars
            if x.dtype.kind in ("U", "S"):
                return "".join(x.flatten().tolist())
            # object array containing strings
            if x.dtype == object:
                try:
                    return str(x.item())
                except Exception:
                    return str(x)
            # numeric array -> string representation
            return str(x.squeeze().tolist())
        return str(x)

    @staticmethod
    def _mat_to_complex_1d(arr) -> np.ndarray:
        """
        Converts MATLAB-loaded 'data' to 1D complex numpy array.
        Supports:
          - complex dtype already
          - real/imag stacked last dim=2
          - purely real
        """
        a = np.asarray(arr)

        if np.iscomplexobj(a):
            x = a.astype(np.complex64, copy=False)
            return x.reshape(-1)

        # If last dim is 2 => [Re, Im]
        if a.ndim >= 1 and a.shape[-1] == 2:
            a = a.astype(np.float32, copy=False)
            re = a[..., 0]
            im = a[..., 1]
            x = re + 1j * im
            return x.astype(np.complex64, copy=False).reshape(-1)

        # otherwise real only
        a = a.astype(np.float32, copy=False)
        return (a + 0j).astype(np.complex64, copy=False).reshape(-1)

    def __getitem__(self, idx):
        cur_filename = self.filenames[idx]
        cur_sample = scio.loadmat(cur_filename, verify_compressed_data_integrity=False)

        for k in ("data", "label", "bits"):
            if k not in cur_sample:
                raise KeyError(f"Missing required key '{k}' in file: {cur_filename}")

        x = self._mat_to_complex_1d(cur_sample["data"])  # np complex64 [L]
        bits = np.asarray(cur_sample["bits"]).reshape(-1)  # np [Lb]
        label = self._mat_to_str(cur_sample["label"])

        return {
            "data": x,      # np.complex64 [L]
            "bits": bits,   # np (numeric) [Lb]
            "label": label  # str
        }


class Collator:
    """
    Produces:
      - data:  [B, N, 1, 2] float32  (real/imag)
      - prompt: list[str]
      - bits:  [B, N] float32        (0/1), aligned with time index

    Notes:
      - N is taken from params.sample_rate (as in your code).
      - input_dim must be 1 for this representation.
      - Per-example normalization is applied to data only (not bits).
    """
    def __init__(self, params, normalize=True):
        self.params = params
        self.normalize = normalize

    def collate(self, minibatch):
        N = int(self.params.sample_rate)     # desired length (e.g., 1024)
        feat_dim = int(self.params.input_dim)
        if feat_dim != 1:
            raise ValueError(f"Expected params.input_dim == 1, got {feat_dim}")

        data_list = []
        bits_list = []
        prompt_list = []

        for record in minibatch:
            # ---------- IQ ----------
            x = record["data"]  # np complex [L] or torch
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            else:
                x = torch.as_tensor(x)

            # ensure complex tensor
            if torch.is_complex(x):
                x_c = x.to(torch.complex64)
            else:
                # allow [L,2] real/imag
                if x.ndim >= 1 and x.shape[-1] == 2:
                    x_f = x.to(torch.float32)
                    x_c = torch.view_as_complex(x_f)
                else:
                    x_c = x.to(torch.float32) + 0j

            x_c = x_c.reshape(-1)  # [L]

            # crop/pad to N
            L = x_c.numel()
            if L < N:
                x_c = F.pad(x_c, (0, N - L))
            elif L > N:
                x_c = x_c[:N]

            # [N] complex -> [N,1] -> [N,1,2] float
            x_c = x_c.view(N, 1)
            x_ri = torch.view_as_real(x_c).to(torch.float32)  # [N,1,2]

            # normalize per-example (optional)
            if self.normalize:
                mean = x_ri.mean()
                std = x_ri.std(unbiased=False)
                if std < 1e-8:
                    std = torch.tensor(1.0, device=x_ri.device)
                x_ri = (x_ri - mean) / std

            data_list.append(x_ri)

            # ---------- prompt ----------
            prompt_list.append(str(record["label"]))

            # ---------- bits (sequence) ----------
            b = record["bits"]
            if isinstance(b, torch.Tensor):
                b_t = b.detach().cpu().numpy().reshape(-1)
            else:
                b_t = np.asarray(b).reshape(-1)

            # numeric + force 0/1
            if not np.issubdtype(b_t.dtype, np.number):
                raise ValueError("bits must be numeric (0/1).")

            # crop/pad to N to align with IQ length
            if b_t.size < N:
                b_t = np.pad(b_t.astype(np.float32), (0, N - b_t.size), mode="constant")
            elif b_t.size > N:
                b_t = b_t[:N].astype(np.float32)
            else:
                b_t = b_t.astype(np.float32)

            # force to 0/1 (handles {-1,+1} too)
            # If your bits are already 0/1, this keeps them the same.
            b_t = (b_t != 0).astype(np.float32)

            bits_list.append(torch.from_numpy(b_t))  # [N]

        data = torch.stack(data_list, dim=0)         # [B,N,1,2]
        bits = torch.stack(bits_list, dim=0)         # [B,N]

        return {
            "data": data,
            "prompt": prompt_list,
            "bits": bits
        }

def from_path(params, is_distributed=False):
    data_dir = params.data_dir
    task_id = params.task_id
    if task_id == 0:
        dataset = WiFiDataset(data_dir)
    elif task_id == 1:
        dataset = SimpleSignalDataset(data_dir)
    else:
        raise ValueError("Unexpected task_id.")
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=params.batch_size,
        collate_fn=Collator(params).collate,
        shuffle=not is_distributed,
        num_workers=8,
        sampler=DistributedSampler(dataset) if is_distributed else None,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True)


def from_path_inference(params):
    cond_dir = params.cond_dir
    task_id = params.task_id
    if task_id == 0:
        dataset = WiFiDataset(cond_dir)
    elif task_id == 1:
        dataset = SimpleSignalDataset(cond_dir)
    else:
        raise ValueError("Unexpected task_id.")
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=params.inference_batch_size,
        collate_fn=Collator(params).collate,
        shuffle=False,
        num_workers=os.cpu_count()
        )