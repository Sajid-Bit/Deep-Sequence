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
        self._encoder_layers = encoder_layers
        self._decoder_layers = decoder_layers
        self._output_sequence_length = output_sequence_length

        






    def _cell_(self):
        pass



