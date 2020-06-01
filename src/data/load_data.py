import pandas as pd

from src.data.Data import Data


def load_drop_empty(train_set_path, test_set_path):
    train = pd.read_csv(train_set_path)
    test = pd.read_csv(test_set_path)
    return Data(train.dropna(), test.dropna())
