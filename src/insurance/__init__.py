import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

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

    print((train.corr() ** 2)["charges"].sort_values(ascending=False)[1:])

    #normal distribution
    train["charges"] = np.log1p(train["charges"])
    #plotting_3_chart(train, 'charges')

    sns.distplot(train['smoker'])

    dc.fixing_skewness(train)
    dc.fixing_skewness(test)
    print("Skewness: " + str(train['charges'].skew()))
    final_features = pd.get_dummies(train).reset_index(drop=True)
    print(final_features.shape)