import pandas as pd
from sklearn.linear_model import LinearRegression
from src.data_processing import data_cleaning
from src.data.load_data import load_drop_empty
from sklearn.model_selection import train_test_split

TARGET_PREDICTION_VALUE = 'charges'


def main():
    data = load_drop_empty()
    data_cleaning.removing_objects(data.train_set)
    data_cleaning.removing_objects(data.test_set)
    ## Saving the target values in y
    y = data.train_set[TARGET_PREDICTION_VALUE].reset_index(drop=True)
    all_data = pd.concat((data.train_set, data.test_set)).reset_index(drop=True)
    print("Skewness: " + str(all_data[TARGET_PREDICTION_VALUE].skew()))
    print("Kurtosis: " + str(all_data[TARGET_PREDICTION_VALUE].kurt()))
    ## Dropping the target variable.
    all_data.drop([TARGET_PREDICTION_VALUE], axis=1, inplace=True)
    data_cleaning.fixing_skewness(all_data)
    final_features = pd.get_dummies(all_data).reset_index(drop=True)
    print(final_features.shape)
    X = final_features.iloc[:len(y), :]
    ## Train test split follows this distinguished code pattern and helps creating train and test set to build machine learning.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=0)

    lin_reg_model = LinearRegression(normalize=True, n_jobs=-1)
    lin_reg_model.fit(X_train, y_train)
    lin_reg_model.predict(X_test)
    accuracy = lin_reg_model.score(X_test, y_test)
    print(accuracy)


if __name__ == "__main__":
    main()
