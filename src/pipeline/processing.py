from src.config.configuration import ConfigurationManager
from src.features.pre_prossing_data import DataProcessing
from src.models.simple_model import Recurrent_Net

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

recurrent_net = Recurrent_Net(input_shape=input_shape, horizone=3, layers=[30, 20, 10], cell_type='lstm', )