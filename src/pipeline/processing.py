from src.config.configuration import ConfigurationManager
from src.features.pre_prossing_data import DataProcessing
from src.models.simple_rnn_model import Recurrent_Net
from src.utils.common import read_yaml
from src.constants import PARAM_FILE_PATH
from keras.regularizers import l2
from keras.optimizers import Adam
from src.utils import metrics

config = ConfigurationManager()

# get the data processing config
config = ConfigurationManager()
get_data_processing = config.get_data_processing()

processing = DataProcessing(config=get_data_processing)

train, val, test = processing.simple_split(0.1)

train_x, train_y = processing.get_rnn_inputs(window_size=10, horizon=3, data=train)
val_x, val_y = processing.get_rnn_inputs(window_size=10, horizon=3, data=val)
test_x, test_y = processing.get_rnn_inputs(window_size=10, horizon=3, data=test)

input_shape = (train_x.shape[1], train_x.shape[2])

parameter = read_yaml(PARAM_FILE_PATH)
# get the RNN parameter
parameter_rnn = dict(units=parameter.recurrent['units'],
                     activation='tanh',
                     dropout=parameter.recurrent['dropout'],
                     kernel_regularizer=l2(parameter.recurrent['l2']),
                     recurrent_regularizer=l2(parameter.recurrent['l2']),
                     kernel_initializer='lecun_uniform',
                     recurrent_initializer='lecun_uniform')

recurrent_net = Recurrent_Net(cell_type='rnn', layers=parameter.recurrent['layers'], cell_params=parameter_rnn)

model = recurrent_net.build_mode(input_shape=input_shape, horizon=3)

model.compile(optimizer=Adam(parameter.recurrent['learning_rate']), loss='mse', metrics=metrics)

model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=10)

pred = model.predict(test_x)
print(pred)