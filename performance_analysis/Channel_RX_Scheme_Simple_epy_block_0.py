import numpy as np
from gnuradio import gr
import os
import csv
import time


class blk(gr.sync_block):
    """
    Bitstream BER + alignment estimator + CSV logger (SNR, BER).

    Inputs:
      in0: TX bits (uchar 0/1)
      in1: RX bits (uchar 0/1)

    Outputs:
      none

    Each update appends TWO values to CSV: snr_db, ber
    Logging happens every log_period seconds (default: 1.0s).
    """

    def __init__(self,
                 window_len=4096,
                 max_lag=2000,

                 csv_path="ber_log.csv",
                 append=True,
                 write_header=True,
                 snr_db=0.0,

                 log_period=1.0   # <-- NEW: seconds between CSV writes
                 ):

        gr.sync_block.__init__(
            self,
            name="ber_and_sync_csv",
            in_sig=[np.uint8, np.uint8],
            out_sig=[]
        )

        self.window_len = int(window_len)
        self.max_lag = int(max_lag)
        self.snr_db = float(snr_db)

        self.log_period = float(log_period)
        self._last_log_time = time.time()

        # CSV config
        self.csv_path = str(csv_path)
        self.append = bool(append)
        self.write_header = bool(write_header)

        # Keep enough history to evaluate +/- lags
        self.keep = self.window_len + 2 * self.max_lag + 1024

        self.tx_hist = np.zeros((0,), dtype=np.uint8)
        self.rx_hist = np.zeros((0,), dtype=np.uint8)

        self.cur_ber = np.float32(1.0)
        self.cur_lag = np.float32(0.0)

        # Prepare CSV file
        mode = "a" if self.append else "w"
        file_exists = os.path.exists(self.csv_path)

        self._csv_file = open(self.csv_path, mode, newline="")
        self._csv_writer = csv.writer(self._csv_file)

        if self.write_header and (not file_exists or not self.append):
            self._csv_writer.writerow(["snr_db", "ber"])
            self._csv_file.flush()

    def _append_hist(self, hist, x):
        if hist.size == 0:
            hist = x.copy()
        else:
            hist = np.concatenate((hist, x))
        if hist.size > self.keep:
            hist = hist[-self.keep:]
        return hist

    def _estimate(self):
        W = self.window_len
        L = self.max_lag

        if self.tx_hist.size < (W + L) or self.rx_hist.size < (W + L):
            return None

        tx = self.tx_hist
        rx = self.rx_hist

        tx_tail = tx[-(W + L):]
        rx_tail = rx[-(W + L):]
        rx_ref = rx_tail[-W:]

        best_err = None
        best_lag = 0

        for lag in range(-L, L + 1):
            if lag >= 0:
                tx_win = tx_tail[-(W + lag): -lag] if lag != 0 else tx_tail[-W:]
                if tx_win.size != W:
                    continue
                err = np.count_nonzero(tx_win ^ rx_ref)
            else:
                k = -lag
                rx_win = rx_tail[-(W + k): -k]
                tx_win = tx_tail[-W:]
                if rx_win.size != W:
                    continue
                err = np.count_nonzero(tx_win ^ rx_win)

            if best_err is None or err < best_err:
                best_err = err
                best_lag = lag

        if best_err is None:
            return None

        ber = float(best_err) / float(W)
        return best_lag, ber

    def work(self, input_items, output_items):
        tx_in = input_items[0].astype(np.uint8, copy=False)
        rx_in = input_items[1].astype(np.uint8, copy=False)

        n = min(len(tx_in), len(rx_in))
        if n == 0:
            return 0

        tx_in = tx_in[:n]
        rx_in = rx_in[:n]

        self.tx_hist = self._append_hist(self.tx_hist, tx_in)
        self.rx_hist = self._append_hist(self.rx_hist, rx_in)

        # Continuously update BER estimate when possible
        est = self._estimate()
        if est is not None:
            lag, ber = est
            self.cur_lag = np.float32(lag)
            self.cur_ber = np.float32(ber)

        # ---- TIME-BASED CSV LOGGING ----
        now = time.time()
        if (now - self._last_log_time) >= self.log_period:
            self._last_log_time = now

            self._csv_writer.writerow([
                f"{self.snr_db:.6f}",
                f"{float(self.cur_ber):.10e}"
            ])
            self._csv_file.flush()

        return n

    def stop(self):
        try:
            self._csv_file.close()
        except Exception:
            pass
        return True

