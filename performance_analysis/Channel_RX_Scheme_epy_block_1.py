# ber_from_stats.py
#
# GNU Radio Embedded Python Block
# - Input: message PMT dict on port "stats"
# - Output: float stream (BER)
#
# Expects PMT dict like:
#   ((ber . 0.478723))

import numpy as np
import pmt
from gnuradio import gr


class blk(gr.sync_block):
    def __init__(self, dict_key="ber", emit_nan_if_missing=True, debug=False):
        gr.sync_block.__init__(
            self,
            name="metric_from_stats",
            in_sig=None,
            out_sig=[np.float32],
        )

        self.dict_key = str(dict_key)
        self.emit_nan_if_missing = bool(emit_nan_if_missing)
        self.debug = bool(debug)

        # Message input
        self.message_port_register_in(pmt.intern("stats"))
        self.set_msg_handler(pmt.intern("stats"), self._handle_stats)

        self._have = False
        self._last = np.float32(np.nan)

    # -------------------------
    # Message handler
    # -------------------------
    def _handle_stats(self, msg):
        try:
            payload = msg

            # Expect a dict directly
            if not pmt.is_dict(payload):
                if self.debug:
                    print("[MEAN_LLR_FROM_STATS] payload is not a dict:", payload)
                if self.emit_nan_if_missing:
                    self._last = np.float32(np.nan)
                    self._have = True
                return

            key = pmt.intern(self.dict_key)
            val = pmt.dict_ref(payload, key, pmt.PMT_NIL)

            if val is pmt.PMT_NIL:
                if self.debug:
                    print(f"[MEAN_LLR_FROM_STATS] key '{self.dict_key}' missing")
                if self.emit_nan_if_missing:
                    self._last = np.float32(np.nan)
                    self._have = True
                return

            # Convert PMT to float
            if pmt.is_real(val):
                ber = float(pmt.to_double(val))
            elif pmt.is_integer(val):
                ber = float(pmt.to_long(val))
            elif pmt.is_bool(val):
                ber = 1.0 if pmt.to_bool(val) else 0.0
            else:
                ber = float(pmt.to_python(val))

            self._last = np.float32(ber)
            self._have = True

            if self.debug:
                print(f"[MEAN_LLR_FROM_STATS] mean_llr={ber:.6e}")

        except Exception as e:
            if self.debug:
                print("[MEAN_LLR_FROM_STATS] exception:", repr(e))
            if self.emit_nan_if_missing:
                self._last = np.float32(np.nan)
                self._have = True

    # -------------------------
    # Stream output
    # -------------------------
    def work(self, input_items, output_items):
        out = output_items[0]
        n = len(out)

        if not self._have:
            out[:] = np.float32(np.nan)
            return n

        out[:] = self._last
        return n

