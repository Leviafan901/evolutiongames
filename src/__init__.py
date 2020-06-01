from src.model_training.train_model import train
from src.data_processing import data_cleaning
from src.data.load_data import load_drop_empty
from src.data_preparation.features_creation import final_sets


def main():
    data = load_drop_empty('../data/train.csv', '../data/test.csv')
    data_cleaning.removing_objects(data)
    X_train, X_test, y_train, y_test = final_sets(data)
    train(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
