#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Not titled yet
# GNU Radio version: 3.10.9.2

from PyQt5 import Qt
from gnuradio import qtgui
import os
import sys
sys.path.append(os.environ.get('GRC_HIER_PATH', os.path.expanduser('~/.grc_gnuradio')))

from gnuradio import analog
from gnuradio import blocks
import pmt
from gnuradio import blocks, gr
from gnuradio import channels
from gnuradio.filter import firdes
from gnuradio import filter
from gnuradio import gr
from gnuradio.fft import window
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import gr, pdu
from gnuradio import ieee80211
from presiso import presiso  # grc-generated hier_block
import Channel_RX_Scheme_epy_block_0 as epy_block_0  # embedded python block
import Channel_RX_Scheme_epy_block_0_0 as epy_block_0_0  # embedded python block
import Channel_RX_Scheme_epy_block_1 as epy_block_1  # embedded python block
import Channel_RX_Scheme_epy_block_1_0 as epy_block_1_0  # embedded python block
import Channel_RX_Scheme_epy_block_2 as epy_block_2  # embedded python block
import sip



class Channel_RX_Scheme(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Not titled yet", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Not titled yet")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "Channel_RX_Scheme")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 40e6
        self.noise_amp = noise_amp = 10

        ##################################################
        # Blocks
        ##################################################

        self.qtgui_number_sink_1_3 = qtgui.number_sink(
            gr.sizeof_float,
            0,
            qtgui.NUM_GRAPH_HORIZ,
            1,
            None # parent
        )
        self.qtgui_number_sink_1_3.set_update_time(0.10)
        self.qtgui_number_sink_1_3.set_title("")

        labels = ['', '', '', '', '',
            '', '', '', '', '']
        units = ['', '', '', '', '',
            '', '', '', '', '']
        colors = [("black", "black"), ("black", "black"), ("black", "black"), ("black", "black"), ("black", "black"),
            ("black", "black"), ("black", "black"), ("black", "black"), ("black", "black"), ("black", "black")]
        factor = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]

        for i in range(1):
            self.qtgui_number_sink_1_3.set_min(i, -1)
            self.qtgui_number_sink_1_3.set_max(i, 1)
            self.qtgui_number_sink_1_3.set_color(i, colors[i][0], colors[i][1])
            if len(labels[i]) == 0:
                self.qtgui_number_sink_1_3.set_label(i, "Data {0}".format(i))
            else:
                self.qtgui_number_sink_1_3.set_label(i, labels[i])
            self.qtgui_number_sink_1_3.set_unit(i, units[i])
            self.qtgui_number_sink_1_3.set_factor(i, factor[i])

        self.qtgui_number_sink_1_3.enable_autoscale(False)
        self._qtgui_number_sink_1_3_win = sip.wrapinstance(self.qtgui_number_sink_1_3.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_number_sink_1_3_win)
        self.qtgui_number_sink_1_2 = qtgui.number_sink(
            gr.sizeof_float,
            0,
            qtgui.NUM_GRAPH_HORIZ,
            1,
            None # parent
        )
        self.qtgui_number_sink_1_2.set_update_time(0.10)
        self.qtgui_number_sink_1_2.set_title("")

        labels = ['', '', '', '', '',
            '', '', '', '', '']
        units = ['', '', '', '', '',
            '', '', '', '', '']
        colors = [("black", "black"), ("black", "black"), ("black", "black"), ("black", "black"), ("black", "black"),
            ("black", "black"), ("black", "black"), ("black", "black"), ("black", "black"), ("black", "black")]
        factor = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]

        for i in range(1):
            self.qtgui_number_sink_1_2.set_min(i, -1)
            self.qtgui_number_sink_1_2.set_max(i, 1)
            self.qtgui_number_sink_1_2.set_color(i, colors[i][0], colors[i][1])
            if len(labels[i]) == 0:
                self.qtgui_number_sink_1_2.set_label(i, "Data {0}".format(i))
            else:
                self.qtgui_number_sink_1_2.set_label(i, labels[i])
            self.qtgui_number_sink_1_2.set_unit(i, units[i])
            self.qtgui_number_sink_1_2.set_factor(i, factor[i])

        self.qtgui_number_sink_1_2.enable_autoscale(False)
        self._qtgui_number_sink_1_2_win = sip.wrapinstance(self.qtgui_number_sink_1_2.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_number_sink_1_2_win)
        self.qtgui_number_sink_1_1 = qtgui.number_sink(
            gr.sizeof_float,
            0,
            qtgui.NUM_GRAPH_HORIZ,
            1,
            None # parent
        )
        self.qtgui_number_sink_1_1.set_update_time(0.10)
        self.qtgui_number_sink_1_1.set_title("")

        labels = ['', '', '', '', '',
            '', '', '', '', '']
        units = ['', '', '', '', '',
            '', '', '', '', '']
        colors = [("black", "black"), ("black", "black"), ("black", "black"), ("black", "black"), ("black", "black"),
            ("black", "black"), ("black", "black"), ("black", "black"), ("black", "black"), ("black", "black")]
        factor = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]

        for i in range(1):
            self.qtgui_number_sink_1_1.set_min(i, -1)
            self.qtgui_number_sink_1_1.set_max(i, 1)
            self.qtgui_number_sink_1_1.set_color(i, colors[i][0], colors[i][1])
            if len(labels[i]) == 0:
                self.qtgui_number_sink_1_1.set_label(i, "Data {0}".format(i))
            else:
                self.qtgui_number_sink_1_1.set_label(i, labels[i])
            self.qtgui_number_sink_1_1.set_unit(i, units[i])
            self.qtgui_number_sink_1_1.set_factor(i, factor[i])

        self.qtgui_number_sink_1_1.enable_autoscale(False)
        self._qtgui_number_sink_1_1_win = sip.wrapinstance(self.qtgui_number_sink_1_1.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_number_sink_1_1_win)
        self.qtgui_number_sink_1_0 = qtgui.number_sink(
            gr.sizeof_float,
            0,
            qtgui.NUM_GRAPH_HORIZ,
            1,
            None # parent
        )
        self.qtgui_number_sink_1_0.set_update_time(0.10)
        self.qtgui_number_sink_1_0.set_title("")

        labels = ['', '', '', '', '',
            '', '', '', '', '']
        units = ['', '', '', '', '',
            '', '', '', '', '']
        colors = [("black", "black"), ("black", "black"), ("black", "black"), ("black", "black"), ("black", "black"),
            ("black", "black"), ("black", "black"), ("black", "black"), ("black", "black"), ("black", "black")]
        factor = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]

        for i in range(1):
            self.qtgui_number_sink_1_0.set_min(i, -1)
            self.qtgui_number_sink_1_0.set_max(i, 1)
            self.qtgui_number_sink_1_0.set_color(i, colors[i][0], colors[i][1])
            if len(labels[i]) == 0:
                self.qtgui_number_sink_1_0.set_label(i, "Data {0}".format(i))
            else:
                self.qtgui_number_sink_1_0.set_label(i, labels[i])
            self.qtgui_number_sink_1_0.set_unit(i, units[i])
            self.qtgui_number_sink_1_0.set_factor(i, factor[i])

        self.qtgui_number_sink_1_0.enable_autoscale(False)
        self._qtgui_number_sink_1_0_win = sip.wrapinstance(self.qtgui_number_sink_1_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_number_sink_1_0_win)
        self.qtgui_number_sink_1 = qtgui.number_sink(
            gr.sizeof_float,
            0,
            qtgui.NUM_GRAPH_HORIZ,
            1,
            None # parent
        )
        self.qtgui_number_sink_1.set_update_time(0.10)
        self.qtgui_number_sink_1.set_title("")

        labels = ['', '', '', '', '',
            '', '', '', '', '']
        units = ['', '', '', '', '',
            '', '', '', '', '']
        colors = [("black", "black"), ("black", "black"), ("black", "black"), ("black", "black"), ("black", "black"),
            ("black", "black"), ("black", "black"), ("black", "black"), ("black", "black"), ("black", "black")]
        factor = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]

        for i in range(1):
            self.qtgui_number_sink_1.set_min(i, -1)
            self.qtgui_number_sink_1.set_max(i, 1)
            self.qtgui_number_sink_1.set_color(i, colors[i][0], colors[i][1])
            if len(labels[i]) == 0:
                self.qtgui_number_sink_1.set_label(i, "Data {0}".format(i))
            else:
                self.qtgui_number_sink_1.set_label(i, labels[i])
            self.qtgui_number_sink_1.set_unit(i, units[i])
            self.qtgui_number_sink_1.set_factor(i, factor[i])

        self.qtgui_number_sink_1.enable_autoscale(False)
        self._qtgui_number_sink_1_win = sip.wrapinstance(self.qtgui_number_sink_1.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_number_sink_1_win)
        self.qtgui_number_sink_0 = qtgui.number_sink(
            gr.sizeof_float,
            0,
            qtgui.NUM_GRAPH_HORIZ,
            1,
            None # parent
        )
        self.qtgui_number_sink_0.set_update_time(0.10)
        self.qtgui_number_sink_0.set_title("")

        labels = ['', '', '', '', '',
            '', '', '', '', '']
        units = ['', '', '', '', '',
            '', '', '', '', '']
        colors = [("black", "black"), ("black", "black"), ("black", "black"), ("black", "black"), ("black", "black"),
            ("black", "black"), ("black", "black"), ("black", "black"), ("black", "black"), ("black", "black")]
        factor = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]

        for i in range(1):
            self.qtgui_number_sink_0.set_min(i, -1)
            self.qtgui_number_sink_0.set_max(i, 1)
            self.qtgui_number_sink_0.set_color(i, colors[i][0], colors[i][1])
            if len(labels[i]) == 0:
                self.qtgui_number_sink_0.set_label(i, "Data {0}".format(i))
            else:
                self.qtgui_number_sink_0.set_label(i, labels[i])
            self.qtgui_number_sink_0.set_unit(i, units[i])
            self.qtgui_number_sink_0.set_factor(i, factor[i])

        self.qtgui_number_sink_0.enable_autoscale(False)
        self._qtgui_number_sink_0_win = sip.wrapinstance(self.qtgui_number_sink_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_number_sink_0_win)
        self.presiso_0_0 = presiso()
        self.presiso_0 = presiso()
        self.pdu_tagged_stream_to_pdu_1_0 = pdu.tagged_stream_to_pdu(gr.types.byte_t, 'ref_payload')
        self.pdu_tagged_stream_to_pdu_1 = pdu.tagged_stream_to_pdu(gr.types.byte_t, 'ref_payload')
        self.pdu_tagged_stream_to_pdu_0_2 = pdu.tagged_stream_to_pdu(gr.types.float_t, 'packet_len')
        self.pdu_tagged_stream_to_pdu_0_1 = pdu.tagged_stream_to_pdu(gr.types.float_t, 'packet_len')
        self.pdu_tagged_stream_to_pdu_0_0 = pdu.tagged_stream_to_pdu(gr.types.float_t, 'packet_len')
        self.pdu_tagged_stream_to_pdu_0 = pdu.tagged_stream_to_pdu(gr.types.float_t, 'packet_len')
        self.low_pass_filter_0_0 = filter.fir_filter_ccf(
            1,
            firdes.low_pass(
                1,
                samp_rate,
                10e6,
                2e6,
                window.WIN_HAMMING,
                6.76))
        self.low_pass_filter_0 = filter.fir_filter_ccf(
            1,
            firdes.low_pass(
                1,
                samp_rate,
                10e6,
                2e6,
                window.WIN_HAMMING,
                6.76))
        self.ieee80211_trigger_0_0 = ieee80211.trigger()
        self.ieee80211_trigger_0 = ieee80211.trigger()
        self.ieee80211_sync_0_0 = ieee80211.sync()
        self.ieee80211_sync_0 = ieee80211.sync()
        self.ieee80211_signal_0_0 = ieee80211.signal()
        self.ieee80211_signal_0 = ieee80211.signal()
        self.ieee80211_demod_0_0 = ieee80211.demod(0, 2)
        self.ieee80211_demod_0 = ieee80211.demod(0, 2)
        self.ieee80211_decode_1 = ieee80211.decode(True)
        self.ieee80211_decode_0 = ieee80211.decode(True)
        self.epy_block_2 = epy_block_2.blk(amp=noise_amp, filename='snr_mean_llr_2curves.csv')
        self.epy_block_1_0 = epy_block_1_0.blk(dict_key="mean_abs_llr", emit_nan_if_missing=False, debug=False)
        self.epy_block_1 = epy_block_1.blk(dict_key="mean_abs_llr", emit_nan_if_missing=False, debug=False)
        self.epy_block_0_0 = epy_block_0_0.blk(snr_in_db=True)
        self.epy_block_0 = epy_block_0.blk(snr_in_db=True)
        self.channels_channel_model_0_0 = channels.channel_model(
            noise_voltage=0.0,
            frequency_offset=0.0,
            epsilon=1.0,
            taps=[1.0],
            noise_seed=0,
            block_tags=False)
        self.channels_channel_model_0 = channels.channel_model(
            noise_voltage=0.0,
            frequency_offset=0.0,
            epsilon=1.0,
            taps=[1.0],
            noise_seed=0,
            block_tags=False)
        self.blocks_stream_to_tagged_stream_4_0 = blocks.stream_to_tagged_stream(gr.sizeof_char, 1, 96, "ref_payload")
        self.blocks_stream_to_tagged_stream_4 = blocks.stream_to_tagged_stream(gr.sizeof_char, 1, 96, "ref_payload")
        self.blocks_stream_to_tagged_stream_3 = blocks.stream_to_tagged_stream(gr.sizeof_float, 1, 1, "packet_len")
        self.blocks_stream_to_tagged_stream_2 = blocks.stream_to_tagged_stream(gr.sizeof_float, 1, 1, "packet_len")
        self.blocks_stream_to_tagged_stream_1 = blocks.stream_to_tagged_stream(gr.sizeof_float, 1, 1, "packet_len")
        self.blocks_stream_to_tagged_stream_0 = blocks.stream_to_tagged_stream(gr.sizeof_float, 1, 1, "packet_len")
        self.blocks_nlog10_ff_0_0 = blocks.nlog10_ff(1, 1, 0)
        self.blocks_nlog10_ff_0 = blocks.nlog10_ff(1, 1, 0)
        self.blocks_multiply_const_vxx_0_0 = blocks.multiply_const_ff(10)
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_ff(10)
        self.blocks_moving_average_xx_0_1 = blocks.moving_average_ff(8192, (1/8192), 4000, 1)
        self.blocks_moving_average_xx_0_0_0 = blocks.moving_average_ff(8192, (1/8192), 4000, 1)
        self.blocks_moving_average_xx_0_0 = blocks.moving_average_ff(8192, (1/8192), 4000, 1)
        self.blocks_moving_average_xx_0 = blocks.moving_average_ff(8192, (1/8192), 4000, 1)
        self.blocks_message_debug_0_0 = blocks.message_debug(True, gr.log_levels.info)
        self.blocks_message_debug_0 = blocks.message_debug(True, gr.log_levels.info)
        self.blocks_keep_one_in_n_3 = blocks.keep_one_in_n(gr.sizeof_float*1, 10000)
        self.blocks_keep_one_in_n_2 = blocks.keep_one_in_n(gr.sizeof_float*1, 10000)
        self.blocks_keep_one_in_n_1 = blocks.keep_one_in_n(gr.sizeof_float*1, 10000)
        self.blocks_keep_one_in_n_0 = blocks.keep_one_in_n(gr.sizeof_float*1, 10000)
        self.blocks_file_source_0_1 = blocks.file_source(gr.sizeof_char*1, '/home/rapcole12/Documents/RFLLM/performance_analysis/ref_payload.bin', True, 0, 0)
        self.blocks_file_source_0_1.set_begin_tag(pmt.PMT_NIL)
        self.blocks_file_source_0_0_0 = blocks.file_source(gr.sizeof_gr_complex*1, '/home/rapcole12/Documents/RFLLM/dataset/wifi/gt.bin', True, 0, 0)
        self.blocks_file_source_0_0_0.set_begin_tag(pmt.PMT_NIL)
        self.blocks_file_source_0_0 = blocks.file_source(gr.sizeof_gr_complex*1, '/home/rapcole12/Documents/RFLLM/dataset/wifi/pred.bin', True, 0, 0)
        self.blocks_file_source_0_0.set_begin_tag(pmt.PMT_NIL)
        self.blocks_file_source_0 = blocks.file_source(gr.sizeof_char*1, '/home/rapcole12/Documents/RFLLM/performance_analysis/ref_payload.bin', True, 0, 0)
        self.blocks_file_source_0.set_begin_tag(pmt.PMT_NIL)
        self.blocks_divide_xx_0_0 = blocks.divide_ff(1)
        self.blocks_divide_xx_0 = blocks.divide_ff(1)
        self.blocks_complex_to_mag_squared_0_1 = blocks.complex_to_mag_squared(1)
        self.blocks_complex_to_mag_squared_0_0_0 = blocks.complex_to_mag_squared(1)
        self.blocks_complex_to_mag_squared_0_0 = blocks.complex_to_mag_squared(1)
        self.blocks_complex_to_mag_squared_0 = blocks.complex_to_mag_squared(1)
        self.blocks_add_xx_0_0 = blocks.add_vcc(1)
        self.blocks_add_xx_0 = blocks.add_vcc(1)
        self.analog_noise_source_x_0_0 = analog.noise_source_c(analog.GR_GAUSSIAN, noise_amp, 0)
        self.analog_noise_source_x_0 = analog.noise_source_c(analog.GR_GAUSSIAN, noise_amp, 0)


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.ieee80211_decode_0, 'out'), (self.blocks_message_debug_0, 'store'))
        self.msg_connect((self.ieee80211_decode_0, 'stats'), (self.epy_block_0, 'in'))
        self.msg_connect((self.ieee80211_decode_0, 'stats'), (self.epy_block_1, 'stats'))
        self.msg_connect((self.ieee80211_decode_1, 'out'), (self.blocks_message_debug_0_0, 'store'))
        self.msg_connect((self.ieee80211_decode_1, 'stats'), (self.epy_block_0_0, 'in'))
        self.msg_connect((self.ieee80211_decode_1, 'stats'), (self.epy_block_1_0, 'stats'))
        self.msg_connect((self.pdu_tagged_stream_to_pdu_0, 'pdus'), (self.epy_block_2, 'npr1'))
        self.msg_connect((self.pdu_tagged_stream_to_pdu_0_0, 'pdus'), (self.epy_block_2, 'npr2'))
        self.msg_connect((self.pdu_tagged_stream_to_pdu_0_1, 'pdus'), (self.epy_block_2, 'metric2'))
        self.msg_connect((self.pdu_tagged_stream_to_pdu_0_2, 'pdus'), (self.epy_block_2, 'metric1'))
        self.msg_connect((self.pdu_tagged_stream_to_pdu_1, 'pdus'), (self.ieee80211_decode_0, 'ref_payload'))
        self.msg_connect((self.pdu_tagged_stream_to_pdu_1_0, 'pdus'), (self.ieee80211_decode_1, 'ref_payload'))
        self.connect((self.analog_noise_source_x_0, 0), (self.blocks_add_xx_0, 1))
        self.connect((self.analog_noise_source_x_0, 0), (self.blocks_complex_to_mag_squared_0_0, 0))
        self.connect((self.analog_noise_source_x_0_0, 0), (self.blocks_add_xx_0_0, 1))
        self.connect((self.analog_noise_source_x_0_0, 0), (self.blocks_complex_to_mag_squared_0_0_0, 0))
        self.connect((self.blocks_add_xx_0, 0), (self.low_pass_filter_0, 0))
        self.connect((self.blocks_add_xx_0_0, 0), (self.low_pass_filter_0_0, 0))
        self.connect((self.blocks_complex_to_mag_squared_0, 0), (self.blocks_moving_average_xx_0, 0))
        self.connect((self.blocks_complex_to_mag_squared_0_0, 0), (self.blocks_moving_average_xx_0_0, 0))
        self.connect((self.blocks_complex_to_mag_squared_0_0_0, 0), (self.blocks_moving_average_xx_0_0_0, 0))
        self.connect((self.blocks_complex_to_mag_squared_0_1, 0), (self.blocks_moving_average_xx_0_1, 0))
        self.connect((self.blocks_divide_xx_0, 0), (self.blocks_nlog10_ff_0, 0))
        self.connect((self.blocks_divide_xx_0_0, 0), (self.blocks_nlog10_ff_0_0, 0))
        self.connect((self.blocks_file_source_0, 0), (self.blocks_stream_to_tagged_stream_4_0, 0))
        self.connect((self.blocks_file_source_0_0, 0), (self.blocks_complex_to_mag_squared_0, 0))
        self.connect((self.blocks_file_source_0_0, 0), (self.channels_channel_model_0, 0))
        self.connect((self.blocks_file_source_0_0_0, 0), (self.blocks_complex_to_mag_squared_0_1, 0))
        self.connect((self.blocks_file_source_0_0_0, 0), (self.channels_channel_model_0_0, 0))
        self.connect((self.blocks_file_source_0_1, 0), (self.blocks_stream_to_tagged_stream_4, 0))
        self.connect((self.blocks_keep_one_in_n_0, 0), (self.blocks_stream_to_tagged_stream_2, 0))
        self.connect((self.blocks_keep_one_in_n_1, 0), (self.blocks_stream_to_tagged_stream_1, 0))
        self.connect((self.blocks_keep_one_in_n_2, 0), (self.blocks_stream_to_tagged_stream_0, 0))
        self.connect((self.blocks_keep_one_in_n_3, 0), (self.blocks_stream_to_tagged_stream_3, 0))
        self.connect((self.blocks_moving_average_xx_0, 0), (self.blocks_divide_xx_0, 1))
        self.connect((self.blocks_moving_average_xx_0_0, 0), (self.blocks_divide_xx_0, 0))
        self.connect((self.blocks_moving_average_xx_0_0_0, 0), (self.blocks_divide_xx_0_0, 0))
        self.connect((self.blocks_moving_average_xx_0_1, 0), (self.blocks_divide_xx_0_0, 1))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.qtgui_number_sink_1, 0))
        self.connect((self.blocks_multiply_const_vxx_0_0, 0), (self.qtgui_number_sink_1_3, 0))
        self.connect((self.blocks_nlog10_ff_0, 0), (self.blocks_multiply_const_vxx_0, 0))
        self.connect((self.blocks_nlog10_ff_0_0, 0), (self.blocks_multiply_const_vxx_0_0, 0))
        self.connect((self.blocks_stream_to_tagged_stream_0, 0), (self.pdu_tagged_stream_to_pdu_0_1, 0))
        self.connect((self.blocks_stream_to_tagged_stream_1, 0), (self.pdu_tagged_stream_to_pdu_0_0, 0))
        self.connect((self.blocks_stream_to_tagged_stream_2, 0), (self.pdu_tagged_stream_to_pdu_0_2, 0))
        self.connect((self.blocks_stream_to_tagged_stream_3, 0), (self.pdu_tagged_stream_to_pdu_0, 0))
        self.connect((self.blocks_stream_to_tagged_stream_4, 0), (self.pdu_tagged_stream_to_pdu_1_0, 0))
        self.connect((self.blocks_stream_to_tagged_stream_4_0, 0), (self.pdu_tagged_stream_to_pdu_1, 0))
        self.connect((self.channels_channel_model_0, 0), (self.blocks_add_xx_0, 0))
        self.connect((self.channels_channel_model_0_0, 0), (self.blocks_add_xx_0_0, 0))
        self.connect((self.epy_block_0, 1), (self.blocks_keep_one_in_n_1, 0))
        self.connect((self.epy_block_0, 0), (self.qtgui_number_sink_0, 0))
        self.connect((self.epy_block_0, 2), (self.qtgui_number_sink_1_0, 0))
        self.connect((self.epy_block_0_0, 1), (self.blocks_keep_one_in_n_3, 0))
        self.connect((self.epy_block_0_0, 2), (self.qtgui_number_sink_1_1, 0))
        self.connect((self.epy_block_0_0, 0), (self.qtgui_number_sink_1_2, 0))
        self.connect((self.epy_block_1, 0), (self.blocks_keep_one_in_n_2, 0))
        self.connect((self.epy_block_1_0, 0), (self.blocks_keep_one_in_n_0, 0))
        self.connect((self.ieee80211_demod_0, 0), (self.ieee80211_decode_0, 0))
        self.connect((self.ieee80211_demod_0_0, 0), (self.ieee80211_decode_1, 0))
        self.connect((self.ieee80211_signal_0, 0), (self.ieee80211_demod_0, 0))
        self.connect((self.ieee80211_signal_0_0, 0), (self.ieee80211_demod_0_0, 0))
        self.connect((self.ieee80211_sync_0, 0), (self.ieee80211_signal_0, 0))
        self.connect((self.ieee80211_sync_0_0, 0), (self.ieee80211_signal_0_0, 0))
        self.connect((self.ieee80211_trigger_0, 0), (self.ieee80211_sync_0, 0))
        self.connect((self.ieee80211_trigger_0_0, 0), (self.ieee80211_sync_0_0, 0))
        self.connect((self.low_pass_filter_0, 0), (self.ieee80211_signal_0, 1))
        self.connect((self.low_pass_filter_0, 0), (self.ieee80211_sync_0, 2))
        self.connect((self.low_pass_filter_0, 0), (self.presiso_0, 0))
        self.connect((self.low_pass_filter_0_0, 0), (self.ieee80211_signal_0_0, 1))
        self.connect((self.low_pass_filter_0_0, 0), (self.ieee80211_sync_0_0, 2))
        self.connect((self.low_pass_filter_0_0, 0), (self.presiso_0_0, 0))
        self.connect((self.presiso_0, 1), (self.ieee80211_sync_0, 1))
        self.connect((self.presiso_0, 0), (self.ieee80211_trigger_0, 0))
        self.connect((self.presiso_0_0, 1), (self.ieee80211_sync_0_0, 1))
        self.connect((self.presiso_0_0, 0), (self.ieee80211_trigger_0_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "Channel_RX_Scheme")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.low_pass_filter_0.set_taps(firdes.low_pass(1, self.samp_rate, 10e6, 2e6, window.WIN_HAMMING, 6.76))
        self.low_pass_filter_0_0.set_taps(firdes.low_pass(1, self.samp_rate, 10e6, 2e6, window.WIN_HAMMING, 6.76))

    def get_noise_amp(self):
        return self.noise_amp

    def set_noise_amp(self, noise_amp):
        self.noise_amp = noise_amp
        self.analog_noise_source_x_0.set_amplitude(self.noise_amp)
        self.analog_noise_source_x_0_0.set_amplitude(self.noise_amp)
        self.epy_block_2.amp = self.noise_amp




def main(top_block_cls=Channel_RX_Scheme, options=None):

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
