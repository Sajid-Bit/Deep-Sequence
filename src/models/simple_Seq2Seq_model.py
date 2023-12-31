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
            input_states = []

            if self.cell == LSTMCell:
                decoder_state_input_c = Input(shape=(unit,))
                input_states = [decoder_state_input_h, decoder_state_input_c]

            decoder_inputs.extend(input_states)

        return decoder_inputs

    def _format_encoder_states(self, encoder_states, use_first=False):
        """
        This method is used to Format the states of encoder block
        to used only the last state of the of  first layer of the encoder

        """
        if self.cell == 'LSTM':
            encoder_states = encoder_states[:2] + [Lambda(lambda x: K.zeros_like(x))(s) for s in encoder_states[2:]]

        else:
            encoder_states = encoder_states[:1] + [Lambda(lambda x: K.zeros_like(x))(s) for s in encoder_states[1:]]

        return encoder_states


class Seq2SeqNet(Seq2Seq):

    def __init__(self, *args, **kwargs):
        self._decoder_yht = None
        self._model = None
        super().__init__(*args, **kwargs)

    def build_model(self, encoder_inputs, decoder_inputs):
        """
        -> encoder_inputs : is 3D Tensor the shape is
        [batch_size, input_sequences, num_features]
        -> decoder_inputs : is 3D Tensor the shape is
        [batch_size, output_sequences, num_features]

        """
        # create the input layers of the encoder and decoders
        encoder_inputs = Input(shape=encoder_inputs, name='Encoder_Inputs')
        decoder_inputs = Input(shape=decoder_inputs, name='Decoder_Inputs')

