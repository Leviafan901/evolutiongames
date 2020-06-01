import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_processing import data_cleaning

TARGET_PREDICTION_VALUE = 'charges'


def final_sets(data):
    ## Saving the target values in to_predict
    to_predict = data.train_set[TARGET_PREDICTION_VALUE].reset_index(drop=True)
    all_data = pd.concat((data.train_set, data.test_set)).reset_index(drop=True)
    ## Dropping the target variable.
    all_data.drop([TARGET_PREDICTION_VALUE], axis=1, inplace=True)
    data_cleaning.fixing_skewness(all_data)
    final_features = pd.get_dummies(all_data).reset_index(drop=True)
    random_data_subsets = final_features.iloc[:len(to_predict), :]
    ## Train test split follows this distinguished code pattern and helps creating train and test sets
    return train_test_split(random_data_subsets, to_predict, test_size=.25, random_state=5)
