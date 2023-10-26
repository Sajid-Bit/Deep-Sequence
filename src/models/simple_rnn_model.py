from keras import Model
from keras.layers import LSTMCell, GRUCell, RNN, SimpleRNNCell, Dense, Input


# create class to wrapper around the RNN, LSTM and GRU classes
# Build the rnn with the given number of layers.

class Recurrent:
    def __init__(self, layers, cell_type, cell_params):
        """
        layers: list <- the number of the of hidden neurons for the i-th layer
        cell_type: str <- lstm, rnn, grn
        cell_params: dict <- A dictionary containing all the parameters for the RNN cell
        """
        self.mode = None
        self.horizon = None
        self.layers = layers
        self.cell_params = cell_params
        if cell_type == 'lstm':
            self.cell = LSTMCell
        elif cell_type == 'gru':
            self.cell = GRUCell
        elif cell_type == 'rnn':
            self.cell = SimpleRNNCell
        else:
            raise NotImplementedError('{0} is not a valid cell type.'.format(cell_type))
        # Build deep rnn
        self.rnn = self._build_()

    # build the RNN model

    def _build_(self):
        cells = []
        for _ in range(self.layers):
            cells.append(self.cell(**self.cell_params))

        # create the RNN
        deep_rnn = RNN(cells, return_sequences=False, return_state=False)

        return deep_rnn

    def build_mode(self):
        pass

    def predict(self, inputs):
        pass

    def evaluate(self):
        pass


class Recurrent_Net(Recurrent):
    """
    Create Recurrent using Multiple Input Multiple Output
    """

    def build_mode(self, input_shape, horizon):
        """
        Create the model with take
        input_shape <- (window_size, n_feature)
        horizon <- int the forecasting step
        The input of the model is :
            - 3D Tensor the shape of the input (batch_size, window_size, n_feature)
        The outputs of the model is:
            -2D Tensor the shape of the output is (batch_size, num_steps)
        """

        self.horizon = horizon
        # create the input of the model
        if len(input_shape) < 2:
            input_shape = (input_shape[0], 1)
        input_layer = Input(shape=input_shape, name='input')
        rnn_model = self.rnn(input_layer)
        outputs_layer = Dense(horizon, activation=None)(rnn_model)

        self.mode = Model(inputs=[input_layer], outputs=[outputs_layer])

        return self.mode

    def predict(self, inputs):
        """
        input <- numpy array (batch_size, window_size, n_feature)
        Return:
            - 2D Tensor (batch_size, horizon)
        """
        return self.mode.predict(inputs)



