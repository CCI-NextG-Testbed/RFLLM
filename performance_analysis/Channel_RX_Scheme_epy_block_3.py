import time
import random
from gnuradio import gr

class blk(gr.basic_block):
    def __init__(self,
                 tb=None,
                 var_name="noise_amp",
                 min_val=0.01,
                 max_val=1.0,
                 period_sec=1.0):

        gr.basic_block.__init__(
            self,
            name="random_noise_controller",
            in_sig=None,
            out_sig=None
        )

        self.tb = tb
        self.var_name = var_name
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.period = float(period_sec)

        self.last_time = 0.0

        print(f"[RAND NOISE] Initialized:")
        print(f"  variable = {self.var_name}")
        print(f"  range    = [{self.min_val}, {self.max_val}]")
        print(f"  period   = {self.period}s")

    def work(self, input_items, output_items):
        now = time.time()

        if now - self.last_time >= self.period:
            val = random.uniform(self.min_val, self.max_val)

            # Call top block setter dynamically
            try:
                setter = getattr(self.tb, f"set_{self.var_name}")
                setter(val)
                print(f"[RAND NOISE] {self.var_name} = {val:.6f}")
            except Exception as e:
                print(f"[RAND NOISE] ERROR calling setter: {e}")

            self.last_time = now

        return 0

