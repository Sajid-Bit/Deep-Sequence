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
                 cell_type='LSTM'):
        self._encoder_layers = encoder_layers
        self._decoder_layers = decoder_layers
        self._output_sequence_length = output_sequence_length

        self.l2 = l2
        if cell_type == 'LSTM':
            self.cell = LSTMCell
        elif cell_type == 'GRN':
            self.cell = GRUCell
        else:
            raise ValueError('{0} is not a valid cell type. Choose between gru and lstm.'.format(cell_type))

        self.encoder = self._build_encoder_()
        self.decoder = self._build_decoder_()

    def _build_encoder(self):
        # create the encoder layers by stacked RNN layers

        encoder = []
        for n_neurons in self._encoder_layers:
            encoder.append(self.cell(units=n_neurons,
                                     dropout=self.dropout,
                                     kernel_regularizer=l2(self.l2),
                                     recurrent_regularizer=l2(self.l2)))

        return RNN(encoder, return_state=True, name='encoder-layers')

    def _build_decoder_(self):
        # create decoder the decoder layers

        decoder = []
        for n_neurons in self._decoder_layers:
            decoder.append(self.cell(units=n_neurons,
                                     dropout=self.dropout,
                                     kernel_regularizer=l2(self.l2),
                                     recurrent_regularizer=l2(self.l2)))

        return RNN(decoder, return_state=True, return_sequences=True, name="decoder-layers")

    def _get_decoder_states(self):
        """
        get the status of the decoder as Input layers
        """
        decoder_inputs = []  # the input status of the decoder
        for unit in self._encoder_layers:
            # take the h status as input lays
            decoder_state_input_h = Input(shape=(unit,))
