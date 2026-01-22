import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from stablediff.diffusion import SignalDiffusion, GaussianDiffusion
from stablediff.dataset import _nested_map

class IQPlusBitsLoss(nn.Module):
    def __init__(self, fft_weight=1.0, bits_weight=0.05):
        super().__init__()
        self.fft_weight = fft_weight
        self.bits_weight = bits_weight

    @staticmethod
    def complex_mse(target_ri, est_ri):
        # target_ri, est_ri: [B,N,1,2] float
        target_c = torch.view_as_complex(target_ri)  # [B,N,1]
        est_c    = torch.view_as_complex(est_ri)
        return torch.mean(torch.abs(target_c - est_c) ** 2)

    def forward(self, target_ri, est_ri, bits=None):
        # IQ loss (time)
        t_loss = self.complex_mse(target_ri, est_ri)

        # IQ loss (freq) along time axis N => dim=1 (still complex)
        target_c = torch.view_as_complex(target_ri)   # [B,N,1]
        est_c    = torch.view_as_complex(est_ri)
        target_fft = torch.fft.fft(target_c, dim=1)
        est_fft    = torch.fft.fft(est_c, dim=1)
        f_loss = torch.mean(torch.abs(target_fft - est_fft) ** 2)

        loss = t_loss + self.fft_weight * f_loss

        # Bits loss (BPSK): use real-part as logits
        if bits is not None:
            # bits: [B,N] float {0,1}
            # est_ri[...,0] is real part: [B,N,1] -> [B,N]
            logits = est_ri[..., 0].squeeze(-1)
            bits_f = bits.to(logits.device).float()

            # If your mapping is reversed (0 -> -1, 1 -> +1), flip logits:
            # logits = -logits
            b_loss = F.binary_cross_entropy_with_logits(logits, bits_f)
            loss = loss + self.bits_weight * b_loss

        return loss
        

class tfdiffLearner:
    def __init__(self, log_dir, model_dir, model, dataset, optimizer, params, *args, **kwargs):
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.task_id = params.task_id
        self.log_dir = log_dir
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.device = model.device
        self.diffusion = SignalDiffusion(params) if params.signal_diffusion else GaussianDiffusion(params)
        # self.prof = torch.profiler.profile(
        #     schedule=torch.profiler.schedule(skip_first=1, wait=0, warmup=2, active=1, repeat=1),
        #     on_trace_ready=torch.profiler.tensorboard_trace_handler(self.log_dir),
        #     with_modules=True, with_flops=True
        # )
        # eeg
        # self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
        #     self.optimizer, 5, gamma=0.5)
        # mimo
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
        )
        self.params = params
        self.iter = 0
        self.is_master = True
        self.loss_fn = IQPlusBitsLoss(fft_weight=1.0, bits_weight=0.05)
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
                tqdm.write(
                    f"\n=== Epoch {epoch_idx} complete === "
                    f"mean_loss={epoch_loss_mean:.6f} over {epoch_loss_count} iters\n"
                )

                # ---- TensorBoard epoch loss ----
                writer = self.summary_writer or SummaryWriter(self.log_dir, purge_step=self.iter)
                writer.add_scalar("train/epoch_loss", epoch_loss_mean, epoch_idx)
                writer.flush()
                self.summary_writer = writer

            self.lr_scheduler.step(epoch_loss_mean)

    def train_iter(self, features):
        self.optimizer.zero_grad()
        data = features['data']          # [B, ...]
        prompts = features['prompt']     # list[str]
        bits = features.get('bits', None) # [B, cond_dim] or None

        B = data.shape[0]
        t = torch.randint(0, self.diffusion.max_step, [B], dtype=torch.int64, device=data.device)

        degrade_data = self.diffusion.degrade_fn(data, t, self.task_id)

        # model must accept prompts as list[str] and embed them internally
        # pass conditioning as a dict to support both prompt (text) and bits
        cond = {'prompt': prompts, 'bits': bits}
        predicted = self.model(degrade_data, t, cond)

        if self.task_id == 3:
            data = data.reshape(-1, 512, 1, 2)

        loss = self.loss_fn(data, predicted, bits=bits)
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
