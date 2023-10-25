from keras.losses import MeanAbsoluteError
from keras.metrics import RootMeanSquaredError
from tensorflow import keras
from keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras import Sequential


class Train:
    def __init__(self, input_shape, epochs=700, patience=50, batch_size=64):
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.INPUT_SHAPE = input_shape

    def simpleRNN(self, units, dropout=0.1):
        # simpleRNN
        model = tf.keras.models.Sequential(
            [SimpleRNN(units=units, return_sequences=False, input_shape=self.INPUT_SHAPE),
             tf.keras.layers.Dropout(dropout),
             tf.keras.layers.Dense(1)
             ])

        return model

    def deep_RNN(self, units, dropout=0.1):
        deep_RNN = tf.keras.models.Sequential([
            tf.keras.layers.SimpleRNN(units=units, return_sequences=True, input_shape=self.INPUT_SHAPE),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.SimpleRNN(units=units, return_sequences=False),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(1)
        ])

        return deep_RNN

    def train_model(self, model, x_train, y_train, x_val, y_val):
        model.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=[RootMeanSquaredError(),
                               MeanAbsoluteError()])

        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=self.patience)

        history = model.fit(x_train, y_train,
                            shuffle=False, epochs=self.epochs,
                            batch_size=self.batch_size,
                            validation_data=(x_val, y_val),
                            callbacks=[es], verbose=1)

        return history

    @staticmethod
    def deepNN(input_shape):
        # Build the model
        model_baseline = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, input_shape=[input_shape], activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1)
        ])

        return model_baseline

    def simple_LSTM(self, input_shape):
        model_tune = tf.keras.models.Sequential([
            tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                                   input_shape=[input_shape]),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True, input_shape=[input_shape])),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Dense(1)
        ])

        return model_tune
