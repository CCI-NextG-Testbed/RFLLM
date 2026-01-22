"""
Embedded Python Blocks:

Each time this file is saved, GRC will instantiate the first class it finds
to get ports and parameters of your block. The arguments to __init__  will
be the parameters. All of them are required to have default values!
"""

import numpy as np
from gnuradio import gr
import time
import pmt
import math

class blk(gr.sync_block):
    def __init__(self, snr_in_db=False):
        gr.sync_block.__init__(self, name="rx_metrics_to_numbers",
                               in_sig=None, out_sig=[np.float32, np.float32, np.float32])

        self.message_port_register_in(pmt.intern("in"))
        self.set_msg_handler(pmt.intern("in"), self.handle_msg)

        self.snr_in_db = bool(snr_in_db)

        # Hold last values (Number Sink will display these)
        self.cfo = np.float32(0.0)
        self.snr = np.float32(0.0)
        self.rssi = np.float32(0.0)

    def _get(self, d, key):
        k = pmt.intern(key)

        # Case 1: PMT dict
        if pmt.is_dict(d) and pmt.dict_has_key(d, k):
            v = pmt.dict_ref(d, k, pmt.PMT_NIL)
            if pmt.is_number(v):
                return float(pmt.to_double(v))

        # Case 2: association list (alist): ((key . val) (key . val) ...)
        cur = d
        while pmt.is_pair(cur):
            item = pmt.car(cur)      # (key . val)
            cur = pmt.cdr(cur)
            if pmt.is_pair(item):
                kk = pmt.car(item)
                vv = pmt.cdr(item)
                if pmt.equal(kk, k) and pmt.is_number(vv):
                    return float(pmt.to_double(vv))

        return None

    def handle_msg(self, msg):
        # If this is a PDU, msg = (meta, data). If not, it might be a dict or an alist.
        d = msg

        if pmt.is_pair(msg):
            meta = pmt.car(msg)
            data = pmt.cdr(msg)
            # Only treat as PDU if meta is a dict (common PDU convention)
            if pmt.is_dict(meta):
                d = meta
            else:
                # Otherwise it's likely an alist like: ((k . v) (k . v) ...)
                d = msg

        # Now extract values from d (dict OR alist)
        cfo = self._get(d, "cfo_hz") or self._get(d, "cfo")
        snr = self._get(d, "snr") or self._get(d, "snr_lin") or self._get(d, "snr_db")
        rssi = self._get(d, "rssi") or self._get(d, "rx_power")

        if cfo is not None:
            self.cfo = np.float32(cfo / 1e3)

        if snr is not None:
            if self.snr_in_db:
                snr_db = 10.0 * math.log10(max(1e-12, snr))  # assumes snr is linear
                self.snr = np.float32(snr_db)
            else:
                self.snr = np.float32(snr)

        if rssi is not None:
            self.rssi = np.float32(rssi)


    def work(self, input_items, output_items):
        # output constant streams = last known values
        output_items[0][:] = self.cfo   # CFO (Hz)
        output_items[1][:] = self.snr   # SNR (linear or dB)
        output_items[2][:] = self.rssi  # RSSI
        return len(output_items[0])

