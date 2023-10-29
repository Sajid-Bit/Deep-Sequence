## this file have the CNN layees
import os
import sys
import shutil
import numpy as np
import pandas as pd
import datetime as dt
from glob import glob
from sklearn.preprocessing import MinMaxScaler
from argparse import ArgumentParser
from keras import regularizers
from keras.models import Model, Sequential
from keras.layers import Dense, Conv1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.optimizers import RMSprop


class CNN_Models:
    def __init__(self, learning_rate,
                 input_shape, latent_dim_1,
                 latent_dim_2, kernel_size,
                 horizon):
        self._learning_rate = learning_rate
        self._input_shape = input_shape
        self._latent_dim_1 = latent_dim_1
        self._latent_dim_2 = latent_dim_2
        self._kernel_size = kernel_size,
        self._horizon = horizon

        self._l2 = 0.001
        self._model = self._build_model_()

    def _build_model_(self):
        """
        Create the model with
        """
        model = Sequential()
        model.add(
            Conv1D(
                self._latent_dim_1,
                kernel_size=self._kernel_size,
                activation='relu',
                dilation_rate=1,
                input_shape=self._input_shape,
                kernel_regularizer=regularizers.l2(self._l2),
                bias_regularizer=regularizers.l2(self._l2)
            )
        )
        if self._latent_dim_2:
            model.add(
                Conv1D(
                    self._latent_dim_1,
                    kernel_size=self._kernel_size,
                    activation='relu',
                    dilation_rate=1,
                    input_shape=self._input_shape,
                    kernel_regularizer=regularizers.l2(self._l2),
                    bias_regularizer=regularizers.l2(self._l2)
                ))
        # create to flatten layer
        model.add(
            Dense(
                self._horizon,
                activation='linear',
                kernel_regularizer=regularizers.l2(self._l2),
                bias_regularizer=regularizers.l2(self._l2)
            )
        )

        optimizer = RMSprop(lr=self._learning_rate)
        model.compile(optimizer=optimizer, loss="mse")

        return model
