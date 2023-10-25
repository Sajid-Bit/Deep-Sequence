import pandas as pd
import numpy as np
from tqdm import tqdm

from src.entity.config_entity import DataProcessingConfig


class DataProcessing:
    def __init__(self, config: DataProcessingConfig):
        self.config = config
        self._data_frame = None
        self._read_data_()

    def _read_data_(self):
        """
        Reade Univariate data of shape (n_samples, n_features)
        : param index_col: the index colume of dataframe
        : param target: the target colume
        """
        data = pd.read_csv(self.config.file_path, index_col=self.config.index_colume)
        self._data_frame = data[[self.config.target]]

    def get_data(self):
        return self._data_frame

    def simple_split(self, test_size=0.1):
        test_len = int(len(self._data_frame) * test_size)
        train_len = len(self._data_frame) - test_len
        val_len = int(test_len * test_size)  # the size of the val take for the test

        return self._data_frame[:train_len], \
            self._data_frame[train_len:train_len + val_len], \
            self._data_frame[train_len + val_len:]

    def get_rnn_inputs(self, window_size, horizon,
                       multivariate_output=False, shuffle=False, other_horizon=None, data=None):
        """
        Prepare data for feeding a RNN model.
        :param data: numpy.array
            shape (n_samples, n_features) or (M, n_samples, n_features)
        :param window_size: int
            Fixed size of the look-back
        :param horizon: int
            Forecasting horizon, the number of future steps that have to be forecasted
        :param multivariate_output: if True, the target array will not have shape
            (n_samples, output_sequence_len) but (n_samples, output_sequence_len, n_features)
        :param shuffle: if True shuffle the data on the first axis
        :param other_horizon:
        :return: tuple
            Return two numpy.arrays: the input and the target for the model.
            the inputs has shape (n_samples, input_sequence_len, n_features)
            the target has shape (n_samples, output_sequence_len)
        """
        if data is None:
            data = self._data_frame
        if data.ndim == 2:
            data = np.expand_dims(data, 0)
        inputs = []
        targets = []
        for X in tqdm(data):  # for each array of shape (n_samples, n_features)
            n_used_samples = X.shape[0] - horizon - window_size + 1
            for i in range(n_used_samples):
                inputs.append(X[i: i + window_size])
                # TARGET FEATURE SHOULD BE THE FIRST
                if multivariate_output:
                    if other_horizon is None:
                        targets.append(
                            X[i + window_size: i + window_size + horizon])
                    else:
                        targets.append(
                            X[i + 1: i + window_size + 1])
                else:
                    if other_horizon is None:
                        targets.append(
                            X[i + window_size: i + window_size + horizon, 0])
                    else:
                        targets.append(
                            X[i + 1: i + window_size + 1, 0])
        encoder_input_data = np.asarray(inputs)  # (n_samples, sequence_len, n_features)
        decoder_target_data = np.asarray(
            targets)  # (n_samples, horizon) or (n_samples, horizon, n_features) if multivariate_output
        idx = np.arange(encoder_input_data.shape[0])
        if shuffle:
            np.random.shuffle(idx)
        return encoder_input_data[idx], decoder_target_data[idx]
