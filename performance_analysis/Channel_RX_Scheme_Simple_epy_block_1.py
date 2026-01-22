import numpy as np
from gnuradio import gr
import math
import os

class blk(gr.sync_block):
    def __init__(self, amp=1.0, fs=1e6, outfile="metrics.csv", period_s=1.0, nv=0.0, snr_db= 0.0):
        gr.sync_block.__init__(
            self,
            name="SNR + EVM CSV Logger",
            in_sig=[np.complex64, np.float32, np.float32],
            out_sig=None
        )
        self.A = float(amp)
        self.fs = float(fs)
        self.outfile = str(outfile)
        self.period_s = float(period_s)
        self.nv = float(nv)
        self.snr_db = float(snr_db)

        # logging cadence in samples
        self.samples_per_log = max(1, int(round(self.fs * self.period_s)))
        self.sample_count = 0

        # EVM accumulators for current 1-second window
        self.win_count = 0
        self.win_sum_e2 = 0.0

        # latest power values (moving-avg streams)
        self.last_Ps = float("nan")
        self.last_Pn = float("nan")

        # create file with header once
        if not os.path.exists(self.outfile):
            with open(self.outfile, "w") as f:
                f.write("noise_voltage,snr_db,evm_rms,evm_mse\n")

    def work(self, input_items, output_items):
        y = input_items[0]      # complex samples (after noise, after Costas if used)
        Ps = input_items[1]     # float stream (moving avg power of signal)
        Pn = input_items[2]     # float stream (moving avg power of noise)

        n = len(y)
        if n == 0:
            return 0

        # Update "latest" power/noise_voltage values from the end of these arrays
        self.last_Ps = float(Ps[-1])
        self.last_Pn = float(Pn[-1])

        # ----- EVM for this chunk -----
        # nearest ideal point (+A or -A) on real axis
        A = self.A
        s_hat = np.where(np.real(y) >= 0.0, A, -A).astype(np.float32).astype(np.complex64)
        e = y - s_hat
        e2 = np.abs(e) ** 2

        self.win_sum_e2 += float(np.sum(e2))
        self.win_count += n

        # ----- Logging cadence -----
        start = self.sample_count
        self.sample_count += n

        # We might cross one or more log boundaries in this call.
        # We'll log once per boundary; for simplicity, log at most once per work call.
        if (start // self.samples_per_log) != ((self.sample_count - 1) // self.samples_per_log):
            # compute metrics for the last window
            evm_mse = self.win_sum_e2 / max(1, self.win_count)
            evm_rms = math.sqrt(evm_mse) / max(1e-12, A)

            Ps_val = self.last_Ps
            Pn_val = self.last_Pn

            t_s = (self.sample_count / self.fs)

            with open(self.outfile, "a") as f:
                f.write(f"{self.nv:.6e},{self.snr_db:.6f},{evm_rms:.6e},{evm_mse:.6e}\n")

            # reset window accumulators for next second
            self.win_sum_e2 = 0.0
            self.win_count = 0

        return n

