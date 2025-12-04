import numpy as np
import os
import random
import torch
import torch.nn.functional as F
import scipy.io as scio
from stablediff.params import AttrDict
from glob import glob
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
        print(f"Evaluating: {cur_filename}")
        cur_sample = scio.loadmat(cur_filename,verify_compressed_data_integrity=False)
        #cur_data = torch.from_numpy(cur_sample['csi_data']).to(torch.complex64)
        cur_data = torch.from_numpy(cur_sample['data']).to(torch.complex64)
        return {
            'data': cur_data,
            'cond': cur_sample['label']     
        }
    
class Collator:
    def __init__(self, params):
        self.params = params

    def collate(self, minibatch):
        N        = self.params.sample_rate   # time length / number of complex samples
        feat_dim = self.params.input_dim     # should be 1
        task_id  = self.params.task_id

        if task_id != 0:
            raise ValueError("Unexpected task_id; this Collator is for WiFi (task_id=0).")

        if feat_dim != 1:
            raise ValueError(f"Expected input_dim=1 for 1×N bits, got {feat_dim}.")

        data_list = []
        cond_list = []

        for record in minibatch:
            x = record["data"]   # IQ for this example

            # ---- Make sure x is a complex torch tensor ----
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            elif not torch.is_tensor(x):
                x = torch.as_tensor(x)

            if torch.is_complex(x):
                x_c = x.to(torch.complex64)
            else:
                # if last dim is 2 -> [Re, Im]
                if x.ndim >= 1 and x.shape[-1] == 2:
                    x_c = torch.view_as_complex(x.to(torch.float32))
                else:
                    # pure real → imag = 0
                    x_c = x.to(torch.float32) + 0j

            # ---- Flatten to 1D complex [L] ----
            x_c = x_c.reshape(-1)
            L = x_c.shape[0]

            # ---- Crop or pad to length N ----
            if L < N:
                pad = N - L
                x_c = F.pad(x_c, (0, pad))        # pad at end with zeros
            elif L > N:
                x_c = x_c[:N]

            # Now x_c is [N] complex
            # ---- Reshape to [N, 1] then to [N, 1, 2] real/imag ----
            x_c = x_c.view(N, 1)                  # [N, 1]
            x_ri = torch.view_as_real(x_c)        # [N, 1, 2]

            # ---- Per-example normalization ----
            mean = x_ri.mean()
            std  = x_ri.std()
            if std < 1e-8:
                std = 1.0
            x_norm = (x_ri - mean) / std          # [N, 1, 2]

            data_list.append(x_norm)

            # ---- cond: keep as pure string prompt ----
            cond = record["cond"]
            if isinstance(cond, np.ndarray):
                cond = cond.item() if cond.shape == () or cond.size == 1 else cond.tolist()
            cond = str(cond)
            cond_list.append(cond)

        if len(data_list) == 0:
            raise RuntimeError("Collator produced an empty batch – check dataset / preprocessing.")

        # Stack: [B, N, 1, 2]
        data = torch.stack(data_list, dim=0)

        return {
            "data": data,      # [B, N, 1, 2]
            "prompt": cond_list  # list[str]
        }
        
def from_path(params, is_distributed=False):
    data_dir = params.data_dir
    task_id = params.task_id
    if task_id == 0:
        dataset = WiFiDataset(data_dir)
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
    else:
        raise ValueError("Unexpected task_id.")
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=params.inference_batch_size,
        collate_fn=Collator(params).collate,
        shuffle=False,
        num_workers=os.cpu_count()
        )
