Deep Sequence
==============================

## Introduction
The package includes several deep learning architectures that can be used for time series forecasting.The package provides also several utilities for **Prossing time Series data and Feature Enginering** e.g <br>

## Project Organization

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   |── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   |    └── visualize.py
    |   |
    │   |── utils <- Scripts to read and w in and worek with file
    |   |    └── common.py
    |   |
    |   |── constants <- File contains constants of the project
    |   |
    |   |── entity <- Scripts to create configuration of all the entity of the project 
    |   |   └── config_entity.py
    |   |
    |   |── pipeline <- Scripts files for the pipeline of prossing, train modeles and testing e.g 
    |
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

 
## Simple Deep Learning Architectures
Example of the dataset will be used to test the Architectures is **ETTh1.csv**
### Example How to Used The RNN Architectures
```python
import numpy as np
from src.models.simple_rnn_models import *
from src.feature.pre_prossing_data import *
from src.featuer import *


```