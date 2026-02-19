import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import csv
from stablediff.diffusion import SignalDiffusion, GaussianDiffusion
from stablediff.dataset import _nested_map

def _init_csv(csv_path="training_log.csv"):
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "mean_loss",
                "lr",
                "attn_type",
            ])

def _append_epoch_csv(epoch, mean_loss, lr, csv_path="training_log.csv", attn_type="ComplexMultiheadAttention"):
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            mean_loss,
            lr,
            attn_type,
        ])


class IQPlusBitsLoss(nn.Module):
    def __init__(self, alpha=0.6, tau=0.1, ema_beta=0.99, eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.tau = tau
        self.ema_beta = ema_beta
        self.eps = eps
        self.ema_iq = None
        self.ema_sym = None

    @staticmethod
    def complex_mse(target_ri, est_ri):
        # target_ri, est_ri: [B,N,1,2] float
        target_c = torch.view_as_complex(target_ri)  # [B,N,1]
        est_c    = torch.view_as_complex(est_ri)
        return torch.mean(torch.abs(target_c - est_c) ** 2)

    @staticmethod
    def _gray_to_binary_t(x: torch.Tensor) -> torch.Tensor:
        b = x.clone()
        shift = 1
        while shift < 32:
            b = torch.bitwise_xor(b, torch.bitwise_right_shift(b, shift))
            shift <<= 1
        return b

    @staticmethod
    def _bits_per_symbol(mod: str) -> int:
        m = str(mod).upper()
        if m == "BPSK":
            return 1
        if m == "QPSK":
            return 2
        if m == "8PSK":
            return 3
        if m == "16QAM":
            return 4
        if m == "64QAM":
            return 6
        if m == "256QAM":
            return 8
        return 1

    def _constellation_points(self, mod: str, device):
        m = str(mod).upper()
        if m == "BPSK":
            return torch.tensor([-1.0 + 0.0j, 1.0 + 0.0j], dtype=torch.complex64, device=device)

        if m == "QPSK":
            return torch.tensor(
                [-1.0 - 1.0j, -1.0 + 1.0j, 1.0 - 1.0j, 1.0 + 1.0j],
                dtype=torch.complex64,
                device=device,
            ) / np.sqrt(2.0)

        if m == "8PSK":
            g = torch.arange(8, dtype=torch.int64, device=device)
            idx = self._gray_to_binary_t(g)
            phase = 2.0 * torch.pi * idx.to(torch.float32) / 8.0
            return torch.exp(1j * phase).to(torch.complex64)

        if m in ("16QAM", "64QAM", "256QAM"):
            M = int(m.replace("QAM", ""))
            k = int(np.log2(M))
            k2 = k // 2
            vals = torch.arange(M, dtype=torch.int64, device=device)

            shifts = torch.arange(k - 1, -1, -1, device=device, dtype=torch.int64)
            bits = ((vals.unsqueeze(1) >> shifts.unsqueeze(0)) & 1).to(torch.int64)  # [M,k]

            w = (2 ** torch.arange(k2 - 1, -1, -1, device=device, dtype=torch.int64))
            gI = (bits[:, :k2] * w.unsqueeze(0)).sum(dim=1)
            gQ = (bits[:, k2:] * w.unsqueeze(0)).sum(dim=1)
            bI = self._gray_to_binary_t(gI)
            bQ = self._gray_to_binary_t(gQ)

            L = int(np.sqrt(M))
            aI = (2.0 * bI.to(torch.float32) - (L - 1)).to(torch.float32)
            aQ = (2.0 * bQ.to(torch.float32) - (L - 1)).to(torch.float32)
            pts = (aI + 1j * aQ).to(torch.complex64)
            p = torch.mean(torch.abs(pts) ** 2).clamp(min=self.eps)
            return pts / torch.sqrt(p)

        # fallback
        return torch.tensor([-1.0 + 0.0j, 1.0 + 0.0j], dtype=torch.complex64, device=device)

    @staticmethod
    def _bits_to_index(bits: torch.Tensor) -> torch.Tensor:
        # bits: [T,k] with MSB-first
        k = bits.shape[1]
        w = (2 ** torch.arange(k - 1, -1, -1, device=bits.device, dtype=torch.float32))
        return torch.sum(bits * w.unsqueeze(0), dim=1).long()

    def _symbol_ce_loss(self, est_c, bits_full, bits_len, modulation, sps):
        # est_c: [B,N] complex
        B, N = est_c.shape
        losses = []
        for i in range(B):
            mod_i = modulation[i]
            k = self._bits_per_symbol(mod_i)
            Li = int(bits_len[i].item())
            Ti = Li // k
            if Ti <= 0:
                continue

            sps_i = max(1, int(sps[i].item()))
            pred_sym = N // sps_i
            T = min(Ti, pred_sym)
            if T <= 0:
                continue

            # predicted symbols at symbol-rate
            est_i = est_c[i, :T * sps_i].view(T, sps_i).mean(dim=1)  # [T]

            # target symbol indices from bits (MSB-first)
            bits_i = bits_full[i, : T * k].view(T, k).to(est_i.device)
            target_idx = self._bits_to_index(bits_i)

            pts = self._constellation_points(mod_i, est_i.device)  # [M]
            d2 = torch.abs(est_i.unsqueeze(1) - pts.unsqueeze(0)) ** 2  # [T,M]
            logits = -d2 / self.tau
            losses.append(F.cross_entropy(logits, target_idx))

        if len(losses) == 0:
            return torch.tensor(0.0, device=est_c.device, dtype=torch.float32)
        return torch.stack(losses).mean()

    def forward(self, target_ri, est_ri, bits_full=None, bits_len=None, modulation=None, sps=None, return_components=False):
        # raw components
        l_iq = self.complex_mse(target_ri, est_ri)

        est_c = torch.view_as_complex(est_ri).squeeze(-1)  # [B,N]
        l_sym = torch.tensor(0.0, device=est_c.device, dtype=torch.float32)
        if bits_full is not None and bits_len is not None and modulation is not None and sps is not None:
            l_sym = self._symbol_ce_loss(est_c, bits_full, bits_len, modulation, sps)

        # EMA normalization for stable blending.
        l_iq_val = float(l_iq.detach().item())
        l_sym_val = float(l_sym.detach().item())
        if self.ema_iq is None:
            self.ema_iq = l_iq_val
        else:
            self.ema_iq = self.ema_beta * self.ema_iq + (1.0 - self.ema_beta) * l_iq_val

        if self.ema_sym is None:
            self.ema_sym = max(l_sym_val, self.eps)
        else:
            self.ema_sym = self.ema_beta * self.ema_sym + (1.0 - self.ema_beta) * max(l_sym_val, self.eps)

        l_iq_n = l_iq / (self.ema_iq + self.eps)
        l_sym_n = l_sym / (self.ema_sym + self.eps)

        if return_components:
            return l_iq_n, l_sym_n
        loss = (1.0 - self.alpha) * l_iq_n + self.alpha * l_sym_n
        return loss
        

class tfdiffLearner:
    def __init__(self, log_dir, model_dir, model, dataset, optimizer, params, *args, **kwargs):
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.device = model.device
        self.diffusion = SignalDiffusion(params) if params.signal_diffusion else GaussianDiffusion(params)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
        )
        self.params = params
        self.iter = 0
        self.is_master = True
        self.loss_fn = IQPlusBitsLoss(
            alpha=float(getattr(params, "loss_alpha", 0.6)),
            tau=float(getattr(params, "symbol_tau", 0.1)),
            ema_beta=float(getattr(params, "loss_ema_beta", 0.99)),
        )
        self.summary_writer = None

    def state_dict(self):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        return {
            'iter': self.iter,
            'model': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items()},
            'optimizer': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items()},
            'params': dict(self.params),
        }

    def load_state_dict(self, state_dict):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            self.model.module.load_state_dict(state_dict['model'])
        else:
            self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.iter = state_dict['iter']

    def save_to_checkpoint(self, filename='weights'):
        save_basename = f'{filename}-{self.iter}.pt'
        save_name = f'{self.model_dir}/{save_basename}'
        link_name = f'{self.model_dir}/{filename}.pt'
        torch.save(self.state_dict(), save_name)
        if os.name == 'nt':
            torch.save(self.state_dict(), link_name)
        else:
            if os.path.islink(link_name):
                os.unlink(link_name)
            os.symlink(save_basename, link_name)

    def restore_from_checkpoint(self, filename='weights'):
        try:
            checkpoint = torch.load(f'{self.model_dir}/{filename}.pt')
            self.load_state_dict(checkpoint)
            return True
        except FileNotFoundError:
            return False

    def train(self, max_iter=None):
        device = next(self.model.parameters()).device
        _init_csv()
        while True:  # epoch
            epoch_loss_sum = 0.0        # <<< NEW
            epoch_loss_count = 0        # <<< NEW

            epoch_idx = self.iter // len(self.dataset)
            iterator = tqdm(self.dataset, desc=f"Epoch {epoch_idx}") if self.is_master else self.dataset

            for features in iterator:
                if max_iter is not None and self.iter >= max_iter:
                    return

                features = _nested_map(
                    features,
                    lambda x: x.to(device) if isinstance(x, torch.Tensor) else x
                )

                loss = self.train_iter(features)

                # -------- loss value extraction --------
                try:
                    loss_val = float(loss.item()) if hasattr(loss, "item") else float(loss)
                except Exception:
                    loss_val = None

                if loss_val is not None:
                    epoch_loss_sum += loss_val        # <<< NEW
                    epoch_loss_count += 1             # <<< NEW

                if torch.isnan(loss).any():
                    raise RuntimeError(f"Detected NaN loss at iteration {self.iter}.")

                # -------- periodic summaries --------
                if self.is_master:
                    if self.iter % 50 == 0 and loss_val is not None:
                        self._write_summary(self.iter, features, loss)

                    if self.iter % len(self.dataset) == 0:
                        self.save_to_checkpoint()

                self.iter += 1

            # ===== END OF EPOCH =====
            if epoch_loss_count > 0:
                epoch_loss_mean = epoch_loss_sum / epoch_loss_count
            else:
                epoch_loss_mean = float("nan")

            
            if self.is_master:
                lr = self.optimizer.param_groups[0].get("lr", float("nan"))

                # ---- CSV logging ----
                _append_epoch_csv(
                    epoch=epoch_idx,
                    mean_loss=epoch_loss_mean,
                    lr=lr
                )

                # ---- checkpoint once per epoch ----
                self.save_to_checkpoint()

                tqdm.write(
                    f"\n=== Epoch {epoch_idx} complete === "
                    f"mean_loss={epoch_loss_mean:.6f}\n"
                )

            # ---- scheduler step ----
            self.lr_scheduler.step(epoch_loss_mean)

    def train_iter(self, features):
        self.optimizer.zero_grad()
        data = features['data']          # [B, ...]
        prompts = features['prompt']     # list[str]
        bits_cond = features.get('bits_cond', features.get('bits', None)) # [B, N] or None
        bits_full = features.get('bits_full', None)
        bits_len = features.get('bits_len', None)
        modulation = features.get('modulation', None)
        sps = features.get('samples_per_symbol', None)

        B = data.shape[0]
        t = torch.randint(0, self.diffusion.max_step, [B], dtype=torch.int64, device=data.device)

        degrade_data = self.diffusion.degrade_fn(data, t)

        # model accepts prompt + symbol-conditioning sequence
        cond = {'prompt': prompts, 'bits_cond': bits_cond}
        predicted = self.model(degrade_data, t, cond)

        l_iq_n, l_sym_n = self.loss_fn(
            data,
            predicted,
            bits_full=bits_full,
            bits_len=bits_len,
            modulation=modulation,
            sps=sps,
            return_components=True,
        )

        # Supervise learned prompt->modulation routing head.
        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        mod_logits = getattr(model_ref, "last_mod_logits", None)
        mod_loss = torch.tensor(0.0, device=data.device, dtype=torch.float32)
        if mod_logits is not None and modulation is not None:
            ids = []
            for m in modulation:
                key = str(m).upper().replace("-", "").replace(" ", "")
                ids.append(model_ref.mod_to_id.get(key, 0))
            target = torch.tensor(ids, dtype=torch.long, device=mod_logits.device)
            mod_loss = F.cross_entropy(mod_logits, target)

        alpha = self.loss_fn.alpha
        loss = (1.0 - alpha) * l_iq_n + 0.5 * alpha * l_sym_n + 0.5 * alpha * mod_loss
        loss.backward()
        self.grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.params.max_grad_norm or 1e9
        )
        self.optimizer.step()
        return loss


    def _write_summary(self, iter, features, loss):
        writer = self.summary_writer or SummaryWriter(self.log_dir, purge_step=iter)
        # writer.add_scalars('feature/csi', features['csi'][0].abs(), step)
        # writer.add_image('feature/stft', features['stft'][0].abs(), step)
        writer.add_scalar('train/loss', loss, iter)
        writer.add_scalar('train/grad_norm', self.grad_norm, iter)
        writer.flush()
        self.summary_writer = writer
