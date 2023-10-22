from src.entity.config_entity import DataIngestionConfig
from src.entity.config_entity import DataProcessingConfig
from src.entity.config_entity import PrepareCallbackConfig
from src.utils.common import read_yaml
from src.constants import CONFIG_FILE_PATH


class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH):
        self.config = read_yaml(config_filepath)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        data_ingestion_config = DataIngestionConfig(
            raw_data_path=config.row_data,
            url_data_path=config.file_data,
            file_name=config.file_data
        )
        return data_ingestion_config

    def get_data_processing(self) -> DataProcessingConfig:
        configs = self.config.data_processing
        data_processing = DataProcessingConfig(
            file_path=configs.file_path,
            window_size=configs.window_size,
            target=configs.target,
            index_colume=configs.index_colume
        )
        return data_processing

    def get_prepare_callback(self) -> PrepareCallbackConfig:
        config = self.config.prepare_callbacks
        prepare_callback = PrepareCallbackConfig(
            save_path_dir=config.root_path_log,
            tensorboard_root_log_dir=config.tensorboard_root_log_dir,
            checkpoint_model_filepath=config.checkpoint_model_filepath
        )
        return prepare_callback


