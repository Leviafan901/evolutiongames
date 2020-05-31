import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import skew
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
#from ploting import distribution as dis_plot

if __name__ == "__main__":
    train = pd.read_csv("../input/train.csv")
    train = train.dropna()
    test = pd.read_csv("../input/test.csv")
    test = test.dropna()

    import data_cleaning as dc
    dc.removing_objects(train)
    dc.removing_objects(test)

    #plotting_3_chart(train, 'charges')
    print("Skewness: " + str(train['charges'].skew()))
    print("Kurtosis: " + str(train['charges'].kurt()))

    ## Saving the target values in "y_train".
    y = train['charges'].reset_index(drop=True)

    all_data = pd.concat((train, test)).reset_index(drop=True)
    ## Dropping the target variable.
    all_data.drop(['charges'], axis=1, inplace=True)

    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)

    print(skewed_feats)

    dc.fixing_skewness(all_data)

    print(all_data.shape)
    final_features = pd.get_dummies(all_data).reset_index(drop=True)
    print(final_features.shape)
    X = final_features.iloc[:len(y), :]
    X_sub = final_features.iloc[len(y):, :]

    from sklearn.model_selection import train_test_split
    ## Train test split follows this distinguished code pattern and helps creating train and test set to build machine learning.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=0)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)