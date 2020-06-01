import pandas as pd

from src.data.Data import Data


def load_drop_empty():
    train = pd.read_csv("../data/train.csv")
    test = pd.read_csv("../data/test.csv")
    return Data(train.dropna(), test.dropna())
