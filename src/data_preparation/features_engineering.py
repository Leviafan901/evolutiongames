import pandas as pd
from src.data_processing import data_cleaning
from sklearn.model_selection import train_test_split


TARGET_PREDICTION_VALUE = 'charges'


def final_sets(data):
    ## Saving the target values in y
    y = data.train_set[TARGET_PREDICTION_VALUE].reset_index(drop=True)
    all_data = pd.concat((data.train_set, data.test_set)).reset_index(drop=True)
    ## Dropping the target variable.
    all_data.drop([TARGET_PREDICTION_VALUE], axis=1, inplace=True)
    data_cleaning.fixing_skewness(all_data)
    final_features = pd.get_dummies(all_data).reset_index(drop=True)
    X = final_features.iloc[:len(y), :]
    ## Train test split follows this distinguished code pattern and helps creating train and test sets
    return train_test_split(X, y, test_size=.33, random_state=0)
