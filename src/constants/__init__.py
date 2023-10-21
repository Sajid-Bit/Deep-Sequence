from pathlib import Path

# create the path for the main configs for the project
# not used in this stub but often useful for finding various files
project_dir = Path().resolve().parents[1]


CONFIG_FILE_PATH = project_dir.joinpath('configs/config.yaml')