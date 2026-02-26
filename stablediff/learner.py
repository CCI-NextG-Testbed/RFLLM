import numpy as np
import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as scio
import matplotlib.pyplot as plt
from matplotlib import animation
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from stablediff.diffusion import SignalDiffusion, GaussianDiffusion
from stablediff.dataset import _nested_map
try:
    from rfml.nn.F import evm as rfml_evm
except Exception:
    rfml_evm = None


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
        self.val_dataset = kwargs.get("val_dataset", None)
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
        )
        self.summary_writer = None
        self.epoch_history = []
        self.snapshot_dir = os.path.join(self.model_dir, "training_snapshots")
        if self.val_dataset is None:
            raise ValueError("A test/validation dataloader is mandatory in this training configuration.")

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
            ckpt_path = f'{self.model_dir}/{filename}.pt'
            try:
                checkpoint = torch.load(ckpt_path, weights_only=True)
            except TypeError:
                checkpoint = torch.load(ckpt_path)
            self.load_state_dict(checkpoint)
            return True
        except FileNotFoundError:
            return False

    @staticmethod
    def _mat_to_str(x):
        if isinstance(x, str):
            return x
        if isinstance(x, (bytes, bytearray)):
            return x.decode("utf-8", errors="ignore")
        arr = np.asarray(x)
        if arr.dtype == object:
            if arr.size == 0:
                return ""
            return tfdiffLearner._mat_to_str(arr.ravel()[0])
        if arr.dtype.kind in ("U", "S"):
            return "".join(arr.ravel().astype(str).tolist())
        if arr.size == 1:
            return str(arr.item())
        return str(arr)

    @staticmethod
    def _normalize_modulation_text(x: str) -> str:
        s = str(x).upper()
        s = "".join(ch for ch in s if ch.isalnum())
        for m in ("BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "256QAM"):
            if m in s:
                return m
        return s

    @staticmethod
    def _to_complex_1d(arr):
        a = np.asarray(arr)
        if np.iscomplexobj(a):
            return a.reshape(-1).astype(np.complex64, copy=False)
        if a.ndim >= 1 and a.shape[-1] == 2:
            return (a[..., 0] + 1j * a[..., 1]).reshape(-1).astype(np.complex64, copy=False)
        return (a.astype(np.float32) + 0j).reshape(-1).astype(np.complex64, copy=False)

    def _load_one_mod_sample(self, data_dir: str, mod: str):
        mats = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(".mat")])
        for p in mats:
            m = scio.loadmat(p, verify_compressed_data_integrity=False)
            if "data" not in m:
                continue
            raw = self._mat_to_str(m["modulation"]) if "modulation" in m else ""
            mod_name = self._normalize_modulation_text(raw)
            if mod_name != mod:
                continue
            x = self._to_complex_1d(m["data"])
            sps = int(np.asarray(m["samples_per_symbol"]).squeeze()) if "samples_per_symbol" in m else 1
            bits = np.asarray(m["bits"]).reshape(-1).astype(np.float32) if "bits" in m else np.zeros((0,), dtype=np.float32)
            bits = (bits != 0).astype(np.float32)
            label = self._mat_to_str(m["label"]) if "label" in m else f"Generate a {mod} signal."
            return {"x": x, "sps": max(1, sps), "bits": bits, "label": str(label), "mod": mod_name}
        raise RuntimeError(f"No sample found for modulation {mod} in {data_dir}")

    @staticmethod
    def _mod_order(mod: str) -> int:
        m = str(mod).upper()
        if m == "BPSK":
            return 2
        if m == "QPSK":
            return 4
        if m == "8PSK":
            return 8
        if m == "16QAM":
            return 16
        if m == "64QAM":
            return 64
        if m == "256QAM":
            return 256
        return 2

    @staticmethod
    def _bits_per_symbol(mod: str) -> int:
        return int(np.log2(tfdiffLearner._mod_order(mod)))

    @staticmethod
    def _bits_to_symbol_index(bits: np.ndarray, k: int) -> np.ndarray:
        if bits.size < k:
            return np.zeros((0,), dtype=np.float32)
        T = bits.size // k
        bb = bits[: T * k].reshape(T, k).astype(np.int64)
        w = (2 ** np.arange(k - 1, -1, -1)).astype(np.int64)
        return (bb * w[None, :]).sum(axis=1).astype(np.float32)

    def _build_bits_cond(self, bits: np.ndarray, mod: str, sps: int, N: int) -> np.ndarray:
        bits = np.asarray(bits).reshape(-1)
        bits = (bits != 0).astype(np.float32)
        k = self._bits_per_symbol(mod)
        M = self._mod_order(mod)
        sym_idx = self._bits_to_symbol_index(bits, k)
        if sym_idx.size == 0:
            bits_cond = np.zeros((N,), dtype=np.float32)
        else:
            if M > 1:
                sym_idx = sym_idx / float(M - 1)
            bits_cond = np.repeat(sym_idx, max(1, int(sps))).astype(np.float32)
            if bits_cond.size < N:
                bits_cond = np.pad(bits_cond, (0, N - bits_cond.size), mode="constant")
            elif bits_cond.size > N:
                bits_cond = bits_cond[:N]
        return bits_cond

    def _prepare_model_input_ri(self, x_c: np.ndarray, N: int, device):
        x = x_c.reshape(-1)
        if x.size < N:
            x = np.pad(x, (0, N - x.size), mode="constant")
        elif x.size > N:
            x = x[:N]
        x_t = torch.from_numpy(x.astype(np.complex64)).to(device).view(N, 1)
        x_ri = torch.view_as_real(x_t).to(torch.float32)  # [N,1,2]
        mean = x_ri.mean()
        std = x_ri.std(unbiased=False)
        if std < 1e-8:
            std = torch.tensor(1.0, device=x_ri.device)
        x_ri = (x_ri - mean) / std
        return x_ri

    def _save_epoch_snapshot(self, epoch_idx: int):
        if not self.is_master:
            return
        if not bool(getattr(self.params, "animate_after_training", False)):
            return
        try:
            data_roots = list(getattr(self.params, "data_dir", []))
            if len(data_roots) == 0:
                return
            data_dir = data_roots[0]
            mods = list(getattr(self.params, "training_animation_mods", ["BPSK", "QPSK", "8PSK"]))
            mods = [str(m).upper() for m in mods][:3]
            N = int(getattr(self.params, "sample_rate", 2048))
            device = next(self.model.parameters()).device

            os.makedirs(self.snapshot_dir, exist_ok=True)
            probe = {m: self._load_one_mod_sample(data_dir, m) for m in mods}

            was_training = self.model.training
            self.model.eval()
            payload = {"epoch": np.array([int(epoch_idx)], dtype=np.int32)}
            with torch.no_grad():
                for m in mods:
                    s = probe[m]
                    x_ri = self._prepare_model_input_ri(s["x"], N=N, device=device)  # [N,1,2]
                    bits_cond = self._build_bits_cond(s["bits"], mod=s["mod"], sps=s["sps"], N=N)
                    bits_t = torch.from_numpy(bits_cond.astype(np.float32)).unsqueeze(0).to(device)

                    x_b = x_ri.unsqueeze(0)  # [1,N,1,2]
                    cond = {"prompt": [s["label"]], "bits_cond": bits_t}
                    pred = self.diffusion.sampling(self.model, cond, device)  # [1,N,1,2]

                    tar_c = torch.view_as_complex(x_b).squeeze(-1).squeeze(0).detach().cpu().numpy().astype(np.complex64)
                    prd_c = torch.view_as_complex(pred).squeeze(-1).squeeze(0).detach().cpu().numpy().astype(np.complex64)

                    payload[f"{m}_target"] = tar_c
                    payload[f"{m}_pred"] = prd_c
                    payload[f"{m}_sps"] = np.array([int(s["sps"])], dtype=np.int32)
            if was_training:
                self.model.train()

            snap_path = os.path.join(self.snapshot_dir, f"snapshot_epoch_{int(epoch_idx):06d}.npz")
            np.savez_compressed(snap_path, **payload)
        except Exception as e:
            print(f"[warn] failed to save epoch snapshot {epoch_idx}: {e}")

    def _evaluate_reverse_diffusion(self):
        if self.val_dataset is None:
            return float("nan")
        device = next(self.model.parameters()).device
        was_training = self.model.training
        self.model.eval()
        loss_sum = 0.0
        loss_count = 0
        with torch.no_grad():
            for features in self.val_dataset:
                features = _nested_map(
                    features,
                    lambda x: x.to(device) if isinstance(x, torch.Tensor) else x
                )

                data = features['data']
                prompts = features['prompt']
                bits_cond = features.get('bits_cond', features.get('bits', None))
                bits_full = features.get('bits_full', None)
                bits_len = features.get('bits_len', None)
                modulation = features.get('modulation', None)
                sps = features.get('samples_per_symbol', None)

                cond = {'prompt': prompts, 'bits_cond': bits_cond}
                predicted = self.diffusion.sampling(self.model, cond, device)

                base_loss, _, _, _ = self.loss_fn(
                    data,
                    predicted,
                    bits_full=bits_full,
                    bits_len=bits_len,
                    modulation=modulation,
                    sps=sps,
                    return_components=True,
                )

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

                mod_loss_weight = float(getattr(self.params, "mod_loss_weight", 0.0))
                loss = base_loss + mod_loss_weight * mod_loss
                loss_sum += float(loss.item())
                loss_count += 1

        if was_training:
            self.model.train()
        if loss_count == 0:
            return float("nan")
        return loss_sum / float(loss_count)

    def _symbol_rate_view(self, x: np.ndarray, sps: int, max_symbols: int):
        T = min(len(x) // sps, max_symbols)
        if T <= 0:
            return np.zeros((0,), dtype=np.complex64)
        return x[: T * sps].reshape(T, sps).mean(axis=1)

    def _generate_training_animation(self):
        if not self.is_master:
            return
        if not bool(getattr(self.params, "animate_after_training", False)):
            return

        try:
            out_path = str(getattr(self.params, "training_animation_out", "./results/training_mods.gif"))
            mods = list(getattr(self.params, "training_animation_mods", ["BPSK", "QPSK", "8PSK"]))
            mods = [str(m).upper() for m in mods][:3]
            wave_samples = int(getattr(self.params, "training_animation_wave_samples", 400))
            wave_stride = int(getattr(self.params, "training_animation_wave_stride", 2))
            max_symbols = int(getattr(self.params, "training_animation_max_symbols", 256))
            fps = int(getattr(self.params, "training_animation_fps", 2))
            snap_files = sorted(glob.glob(os.path.join(self.snapshot_dir, "snapshot_epoch_*.npz")))
            if len(snap_files) == 0:
                print("[warn] animation skipped: no epoch snapshots found")
                return
            snapshots = [np.load(p, allow_pickle=True) for p in snap_files]
            epochs = np.asarray([int(s["epoch"][0]) for s in snapshots], dtype=np.int64)

            fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
            ax_const = [axes[0, 0], axes[0, 1], axes[0, 2]]
            ax_wave = [axes[1, 0], axes[1, 1], axes[1, 2]]
            pred_scats = []
            pred_lines = []
            for i, mod in enumerate(mods):
                s0 = snapshots[0]
                target = np.asarray(s0[f"{mod}_target"])
                pred = np.asarray(s0[f"{mod}_pred"])
                sps = int(np.asarray(s0[f"{mod}_sps"]).squeeze())

                target_sym = self._symbol_rate_view(target, sps=sps, max_symbols=max_symbols)
                pred_sym = self._symbol_rate_view(pred, sps=sps, max_symbols=max_symbols)
                ax_const[i].scatter(
                    np.real(target_sym), np.imag(target_sym),
                    s=10, c="gray", alpha=0.45, label="Tx"
                )
                pred_sc = ax_const[i].scatter(np.real(pred_sym), np.imag(pred_sym), s=10, c="tab:orange", alpha=0.9, label="Pred")
                pred_scats.append((pred_sc, sps))

                ax_const[i].axhline(0, color="k", lw=0.6, ls="--")
                ax_const[i].axvline(0, color="k", lw=0.6, ls="--")
                ax_const[i].set_title(f"{mod} Symbols")
                ax_const[i].set_xlabel("I")
                ax_const[i].set_ylabel("Q")
                ax_const[i].set_aspect("equal", adjustable="box")
                ax_const[i].set_xlim(-2.0, 2.0)
                ax_const[i].set_ylim(-2.0, 2.0)
                ax_const[i].grid(True, alpha=0.3)
                ax_const[i].legend(loc="upper right", fontsize=8)

                n = min(wave_samples, len(target), len(pred))
                stride = max(1, wave_stride)
                t = np.arange(0, n, stride)
                tx_r_vals = np.real(target[:n])[::stride]
                tx_i_vals = np.imag(target[:n])[::stride]
                pr_r_vals = np.real(pred[:n])[::stride]
                pr_i_vals = np.imag(pred[:n])[::stride]

                tx_r, = ax_wave[i].plot(t, tx_r_vals, lw=0.9, ls="--", color="tab:blue", alpha=0.4, label="Tx Real")
                tx_i, = ax_wave[i].plot(t, tx_i_vals, lw=0.9, ls="--", color="tab:orange", alpha=0.4, label="Tx Imag")
                pr_r, = ax_wave[i].plot(t, pr_r_vals, lw=2.0, marker="o", ms=2.2, markevery=max(1, len(t)//60), color="tab:blue", label="Pred Real")
                pr_i, = ax_wave[i].plot(t, pr_i_vals, lw=2.0, marker="o", ms=2.2, markevery=max(1, len(t)//60), color="tab:orange", label="Pred Imag")
                pred_lines.append((pr_r, pr_i, n, stride, sps))

                ax_wave[i].set_title(f"{mod} Signal")
                ax_wave[i].set_xlabel("Sample")
                ax_wave[i].set_ylabel("Amplitude")
                ymax = float(np.max(np.abs(np.concatenate([tx_r_vals, tx_i_vals, pr_r_vals, pr_i_vals])))) if n > 0 else 1.0
                ymax = max(1.0, 1.15 * ymax)
                ax_wave[i].set_ylim(-ymax, ymax)
                ax_wave[i].set_xlim(0, max(1, n - 1))
                ax_wave[i].grid(True, alpha=0.3)
                ax_wave[i].legend(loc="upper right", fontsize=7)

            title = fig.suptitle("", fontsize=12)

            def _update(k):
                e = epochs[k]
                s = snapshots[k]
                for i, mod in enumerate(mods):
                    pred = np.asarray(s[f"{mod}_pred"])
                    pred_sc, sps = pred_scats[i]
                    pred_sym = self._symbol_rate_view(pred, sps=sps, max_symbols=max_symbols)
                    pts = np.column_stack([np.real(pred_sym), np.imag(pred_sym)]) if len(pred_sym) else np.zeros((0, 2))
                    pred_sc.set_offsets(pts)

                    pr_r, pr_i, n, stride, _ = pred_lines[i]
                    n = min(n, len(pred))
                    t = np.arange(0, n, stride)
                    pr_r_vals = np.real(pred[:n])[::stride]
                    pr_i_vals = np.imag(pred[:n])[::stride]
                    pr_r.set_data(t, pr_r_vals)
                    pr_i.set_data(t, pr_i_vals)
                title.set_text(f"Epoch {int(e)}")
                artists = [title]
                artists.extend([x[0] for x in pred_scats])
                artists.extend([x[0] for x in pred_lines])
                artists.extend([x[1] for x in pred_lines])
                return artists

            ani = animation.FuncAnimation(
                fig,
                _update,
                frames=len(epochs),
                interval=max(1, int(1000 / max(1, fps))),
                blit=False,
                repeat=True,
            )

            os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
            if out_path.lower().endswith(".gif"):
                ani.save(out_path, writer=animation.PillowWriter(fps=fps))
            else:
                ani.save(out_path, writer="ffmpeg", fps=fps)
            plt.close(fig)
            print(f"Saved training animation: {out_path}")
        except Exception as e:
            print(f"[warn] failed to generate training animation: {e}")

    def train(self, max_iter=None):
        device = next(self.model.parameters()).device
        stop_training = False
        while True:  # epoch
            epoch_loss_sum = 0.0        # <<< NEW
            epoch_loss_count = 0        # <<< NEW

            epoch_idx = self.iter // len(self.dataset)
            iterator = tqdm(self.dataset, desc=f"Epoch {epoch_idx}") if self.is_master else self.dataset

            for features in iterator:
                if max_iter is not None and self.iter >= max_iter:
                    stop_training = True
                    break

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
            val_loss = float("nan")

            
            if self.is_master:
                # ---- checkpoint once per epoch ----
                self.save_to_checkpoint()
                val_loss = self._evaluate_reverse_diffusion()

                tqdm.write(
                    f"\n=== Epoch {epoch_idx} complete === "
                    f"train_loss={epoch_loss_mean:.6f} "
                    f"test_loss={val_loss:.6f}\n"
                )
            self.epoch_history.append(int(epoch_idx))

            # ---- scheduler step ----
            step_metric = val_loss if self.is_master else epoch_loss_mean
            self.lr_scheduler.step(step_metric)
            if self.is_master and bool(getattr(self.params, "animate_after_training", False)):
                self._save_epoch_snapshot(epoch_idx)
            if stop_training:
                break

        self._generate_training_animation()
        return

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

        base_loss, _, _, _ = self.loss_fn(
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

        mod_loss_weight = float(getattr(self.params, "mod_loss_weight", 0.0))
        loss = base_loss + mod_loss_weight * mod_loss
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
