import pmt
from gnuradio import gr

class blk(gr.basic_block):
    """
    Event-driven stepper (NO threads).
    On each incoming tick message at port 'in', publishes a PMT pair:
        (symbol(varname), value)
    compatible with Message Pair to Var.

    Wiring:
      Message Strobe (out) -> this_block.in
      this_block.out -> Message Pair to Var.inpair
    """

    def __init__(self, varname="snr_db_range", start_val=-20.0, stop_val=7.0, step=1.0, publish_start=True):
        gr.basic_block.__init__(self, name="stepper_msg_pair_no_threads", in_sig=None, out_sig=None)

        self.varname = str(varname)
        self.start_val = float(start_val)
        self.stop_val = float(stop_val)
        self.step = float(step)

        # current value
        self.val = self.start_val

        # ports
        self.message_port_register_in(pmt.intern("in"))
        self.set_msg_handler(pmt.intern("in"), self._on_tick)
        self.message_port_register_out(pmt.intern("out"))

        # behavior
        self.publish_start = bool(publish_start)

    def _publish(self):
        key = pmt.intern(self.varname)
        msg = pmt.cons(key, pmt.from_double(self.val))  # change to from_long(int(self.val)) if you want ints
        self.message_port_pub(pmt.intern("out"), msg)

    def _advance(self):
        self.val += self.step
        if self.val > self.stop_val:
            self.val = self.start_val

    def _on_tick(self, msg):
        if self.publish_start:
            self._publish()
            self._advance()
        else:
            self._advance()
            self._publish()

