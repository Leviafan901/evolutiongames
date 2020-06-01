import pandas as pd
from sklearn.linear_model import LinearRegression
from src.data_processing import data_cleaning
from sklearn.model_selection import train_test_split


def main():
    train = pd.read_csv("../data/train.csv")
    train = train.dropna()
    test = pd.read_csv("../data/test.csv")
    test = test.dropna()

    data_cleaning.removing_objects(train)
    data_cleaning.removing_objects(test)

    print("Skewness: " + str(train['charges'].skew()))
    print("Kurtosis: " + str(train['charges'].kurt()))

    ## Saving the target values in y
    y = train['charges'].reset_index(drop=True)
    all_data = pd.concat((train, test)).reset_index(drop=True)
    ## Dropping the target variable.
    all_data.drop(['charges'], axis=1, inplace=True)
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
