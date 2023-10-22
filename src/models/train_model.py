from src.models.prepare_callback import PrepareCallback
from src.config.configuration import ConfigurationManager

# create the config for the call the PrepareCallback
get_prepare_config = ConfigurationManager.get_prepare_callback()

prepareCallback = PrepareCallback(config=get_prepare_config)



