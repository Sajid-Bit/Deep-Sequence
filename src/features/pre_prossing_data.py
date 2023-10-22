import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.entity.config_entity import DataProcessingConfig


def split_data(df, test_split=0.1):
    n = int(len(df) * test_split)

    train, test = df[:-n], df[-n:]
    return train, test


class Standardize:
    def __init__(self, split=0.15):
        self.sigma = None
        self.mu = None
        self.split = split

    def _transform(self, df):
        return (df - self.mu) / self.sigma

    def fit_transform(self, train, test):
        self.mu = train.mean()

        self.sigma = train.std()
        train_s = self._transform(train)
        test_s = self._transform(test)
        return train_s, test_s

    def transform(self, df):
        return self._transform(df)

    def inverse(self, df):
        return (df * self.sigma) + self.mu

    def inverse_y(self, df):
        return (df * self.sigma[0]) + self.mu[0]


class DataProcessing(StandardScaler):
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

    def simple_split(self, X=None, train_len=None, test_len=None, valid_len=None):
        """
        Split the data in train-test-validation using the given dimensions for each set.
        :param X: numpy.array or pandas.DataFrame
            Univariate data of shape (n_samples, n_features)
        :param train_len: int
            Length in number of data points (measurements) for training.
            If None then allow_muliple_split cannot be True.
        :param test_len: int
            Length in number of data points (measurements) for testing
        :param valid_len: int
            Length in number of data points (measurements) for validation
        :return: list
            train: numpy.array, shape=(train_len, n_features)
            validation: numpy.array, shape=(valid_len, n_features)
            test: numpy.array, shape=(test_len, n_features)
        """
        if X is None:
            X = np.array(self._read_data_())  # select the data that pass to the class
        if test_len is None:
            raise ValueError('test_len cannot be None.')
        if train_len is None:
            train_len = X.shape[0] - test_len
            valid_len = 0
        if valid_len is None:
            valid_len = X.shape[0] - train_len - test_len
        return X[:train_len], \
            X[train_len:train_len + valid_len], \
            X[train_len + valid_len:]
