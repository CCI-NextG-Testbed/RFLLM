import os
import time
import numpy as np
import pmt
from gnuradio import gr


class blk(gr.basic_block):
    def __init__(self, amp=0, filename="snr_mean_llr_log.csv"):
        gr.basic_block.__init__(
            self,
            name="npr_metric_logger",
            in_sig=None,
            out_sig=None,
        )

        # -------------------------
        # Message ports
        # -------------------------
        for port in ("npr1", "metric1", "npr2", "metric2"):
            self.message_port_register_in(pmt.intern(port))

        self.set_msg_handler(pmt.intern("npr1"), self.h_npr1)
        self.set_msg_handler(pmt.intern("metric1"), self.h_ber1)
        self.set_msg_handler(pmt.intern("npr2"), self.h_npr2)
        self.set_msg_handler(pmt.intern("metric2"), self.h_ber2)

        self.amp = amp

        # Latest cached values
        self.latest_npr1 = None
        self.latest_npr2 = None
        self.latest_ber1 = None
        self.latest_ber2 = None

        self.last_write = time.time()
        self.write_period = 2.0  # seconds

        # -------------------------
        # File setup (append mode)
        # -------------------------
        self.filename = os.path.abspath(filename)
        file_exists = os.path.exists(self.filename)

        self.f = open(self.filename, "a", buffering=1)

        if not file_exists:
            self.f.write("snr1_db,metric1,snr2_db,metric2,amp\n")
            self.f.flush()

        print(f"[LOGGER] Appending to CSV: {self.filename}")

    # -------------------------
    # Message handlers
    # -------------------------
    def h_npr1(self, msg):
        self.latest_npr1 = self._to_float(msg)

    def h_npr2(self, msg):
        self.latest_npr2 = self._to_float(msg)

    def h_ber1(self, msg):
        self.latest_ber1 = self._to_float(msg)
        self._maybe_write()

    def h_ber2(self, msg):
        self.latest_ber2 = self._to_float(msg)
        self._maybe_write()

    # -------------------------
    # Core writer
    # -------------------------
    def _maybe_write(self):
        now = time.time()

        # Rate limit
        if now - self.last_write < self.write_period:
            return

        # Need NPRs at minimum
        if self.latest_npr1 is None or self.latest_npr2 is None:
            return

        b1 = self.latest_ber1 if self.latest_ber1 is not None else float("nan")
        b2 = self.latest_ber2 if self.latest_ber2 is not None else float("nan")

        self.f.write(
            f"{self.latest_npr1:.6f},{b1:.8e},"
            f"{self.latest_npr2:.6f},{b2:.8e},{self.amp}\n"
        )
        self.f.flush()
        os.fsync(self.f.fileno())

        print(
            f"[LOGGER] t={now:.1f} "
            f"NPR1={self.latest_npr1:.2f} "
            f"Metric1={b1:.2e}"
        )

        self.last_write = now

    # -------------------------
    # Helpers
    # -------------------------
    def _to_float(self, msg):
        payload = pmt.cdr(msg) if pmt.is_pair(msg) else msg

        if pmt.is_f32vector(payload):
            return float(pmt.f32vector_elements(payload)[0])

        try:
            v = pmt.to_python(payload)
            if isinstance(v, (list, tuple)) and len(v):
                return float(v[0])
            if isinstance(v, (int, float)):
                return float(v)
        except Exception:
            pass

        return None

    def stop(self):
        try:
            self.f.flush()
            os.fsync(self.f.fileno())
            self.f.close()
            print(f"[LOGGER] Closed CSV: {self.filename}")
        except Exception:
            pass
        return True

