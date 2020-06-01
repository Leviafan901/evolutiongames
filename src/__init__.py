from src.data_processing.data_cleaning import convert_predict_value_to_int
from src.model_training.train import linear_regression
from src.data_processing import data_cleaning
from src.data.load_data import load_drop_empty
from src.data_preparation.features_engineering import final_sets


def main():
    data = data_cleaning.removing_objects(load_drop_empty('../data/train.csv', '../data/test.csv'))
    cleaned_data = convert_predict_value_to_int(data)
    X_train, X_test, y_train, y_test = final_sets(cleaned_data)
    linear_regression(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
