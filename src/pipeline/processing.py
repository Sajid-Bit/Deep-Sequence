from src.config.configuration import ConfigurationManager
from src.features.pre_prossing_data import DataProcessing
from src.models.simple_model import Recurrent_Net
from src.utils.common import read_yaml
from src.constants import PARAM_FILE_PATH, CONFIG_FILE_PATH

config = ConfigurationManager()

# get the data processing config
config = ConfigurationManager()
get_data_processing = config.get_data_processing()

processing = DataProcessing(config=get_data_processing)

train, val, test = processing.simple_split(0.1)

train_x, train_y = processing.get_rnn_inputs(window_size=10, horizon=3, data=train)
val_x, val_y = processing.get_rnn_inputs(window_size=10, horizon=3, data=val)
test_x, test_x = processing.get_rnn_inputs(window_size=10, horizon=3, data=test)

input_shape = (train_x.shape[1], train_x.shape[2])

parameter = read_yaml(PARAM_FILE_PATH)
# get the RNN parameter
parameter_rnn = dict(epochs=parameter.recurrent['epochs'],
                     batch_size=parameter.recurrent['batch_size'],
                     dropout=parameter.recurrent['dropout'],
                     units=parameter.recurrent['units'],
                     learning_rate=parameter.recurrent['learning_rate'])

# recurrent_net = Recurrent_Net(cell_type='rnn', layers=parameter['layers'], cell_params=parameter_rnn)
#
# model = recurrent_net.build_mode(input_shape=input_shape, horizon=3)
#
# print(model.summary())
#
