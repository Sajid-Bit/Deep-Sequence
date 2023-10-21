import pandas as pd
import statsmodels
from sklearn.preprocessing import StandardScaler

from src.config.configuration import ConfigurationManager
from src.entity.config_entity import DataProcessingConfig


class Standardize:
    def __init__(self, split=0.15):
        self.sigma = None
        self.mu = None
        self.split = split

    def _transform(self, df):
        return (df - self.mu) / self.sigma

    def split_data(self, df, test_split=0.1):
        n = int(len(df) * test_split)

        train, test = df[:-n], df[-n:]
        return train, test

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
        # read the file as
        self._data_frame = pd.read_csv(self.config.file_path)

    def get_data(self):
        return self._data_frame

    def sort_time_series(self):
        """
        Parse the datetime field, Sort the values accordingly and save the new dataframe to disk
        save the new dataframe in processed data dir
        """
        # select the data colume
        index = 'date'

        return index
