from src.config.configuration import ConfigurationManager
from src.features.pre_prossing_data import DataProcessing

# git the configuration data
config = ConfigurationManager()
config_processing = config.get_data_processing()

data_processing = DataProcessing(config_processing)

data = data_processing.get_data()

index = data_processing.sort_time_series()


