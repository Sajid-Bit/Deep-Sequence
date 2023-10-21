from pathlib import Path
from src.entity.config_entity import DataIngestionConfig
import os
import gdown


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            gdown.download(url=self.config.source_URL, output=self.config.local_data_file, quiet=True)
            filename = os.path.split(self.config.local_data_file)[-1]

        else:
            print('This file is exist')
