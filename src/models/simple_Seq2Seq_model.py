from keras.layers import Concatenate, LSTMCell, GRU, GRUCell, RNN, Dense, Input, Lambda, TimeDistributed, Dropout
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from keras import Model
import keras.backend as K
import numpy as np
from itertools import chain
import keras
from keras.regularizers import l2
from keras.metrics import get
import tensorflow as tf


class Seq2Seq:
    def __init__(self, encoder_layers,
                 decoder_layers,
                 output_sequence_length,
                 units=30,
                 dropout=0.0,
                 l2=0.01,
                 cell_type='lstm'):
        self.dropout = dropout
        self.output_sequence_length = output_sequence_length  # the number of the output step to predict
        self.decoder_layers = decoder_layers
        self.encoder_layers = encoder_layers
        self.l2 = l2
        self._cell_type = cell_type
        self._units = None
        self.cell = self._cell_()
        self.models = None

    # RNN with the layer of cells
    def _cell_(self):
        if self.__cell_type == "LSTM":
            cell = tf.nn.rnn_cell.LSTMCell()
        elif self.__cell_type == "GRU":
            cell = tf.nn.rnn_cell.GRUCell()
        elif self.__cell_type == "RNN":
            cell = tf.nn.rnn_cell.BasicRNNCell()
        return cell

    def bulid_model(self, units=30):
        pass
