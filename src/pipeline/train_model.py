from src.models.simple_rnn_model import Recurrent_Net
# Generate synthetic data
import numpy as np
import pandas as pd
n_features = 10
X = np.random.uniform(0., 10., size=(10000, n_features))

data_frame = pd.DataFrame(X, columns=[f'X_{i}' for i in range(1, 11)])


