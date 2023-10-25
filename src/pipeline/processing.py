from src.config.configuration import ConfigurationManager
from src.features.pre_prossing_data import DataProcessing
from src.models.predict_model import Train

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

train_model = Train(input_shape=input_shape, epochs=10)

simple_Rnn = train_model.simpleRNN(units=30)

mode = train_model.train_model(simple_Rnn, train_x, train_y, val_x, val_y)