import numpy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src import load_drop_empty
from src.data_preparation.features_engineering import TARGET_PREDICTION_VALUE
from src.data_processing import data_cleaning
from src.data_processing.data_cleaning import removing_objects
from src.visualization.distribution import normal_distribution_chart
from src.visualization.plots import customized_scatterplot, linearity_regplot, linearity_residplot, all_features_heatmap


def main():
    data = load_drop_empty('../../data/train.csv', '../../data/test.csv')
    print(data.train_set.shape)
    print(data.test_set.shape)
    removing_objects(data)
    print((data.train_set.corr() ** 2)[TARGET_PREDICTION_VALUE].sort_values(ascending=False)[1:])
    normal_distribution_chart(data.train_set, TARGET_PREDICTION_VALUE)
    customized_scatterplot(data.train_set.charges, data.train_set.smoker, 'smoker')
    linearity_residplot(data.train_set.smoker, data.train_set.charges, 'smoker')
    linearity_residplot(data.train_set.bmi, data.train_set.charges, 'bmi')
    customized_scatterplot(data.train_set.charges, data.train_set.age, 'age')
    linearity_residplot(data.train_set.age, data.train_set.charges, 'age')
    print("Skewness: " + str(data.test_set[TARGET_PREDICTION_VALUE].skew()))
    print("Kurtosis: " + str(data.test_set[TARGET_PREDICTION_VALUE].kurt()))
    previous_data = data
    # improving the normal distribution
    data.train_set[TARGET_PREDICTION_VALUE] = numpy.log1p(data.train_set[TARGET_PREDICTION_VALUE])
    linearity_residplot(previous_data.train_set.age, previous_data.train_set.charges, 'age')
    linearity_residplot(data.train_set.age, data.train_set.charges, 'age')
    linearity_residplot(data.train_set.bmi, data.train_set.charges, 'bmi')
    normal_distribution_chart(data.train_set, TARGET_PREDICTION_VALUE)
    all_features_heatmap(data.train_set)
    all_data = pd.concat((data.train_set, data.test_set)).reset_index(drop=True)
    all_data.drop([TARGET_PREDICTION_VALUE], axis=1, inplace=True)
    plt.show(sns.distplot(all_data['bmi']))
    plt.show(sns.distplot(all_data['age']))
    data_cleaning.fixing_skewness(all_data)
    plt.show(sns.distplot(all_data['bmi']))
    plt.show(sns.distplot(all_data['age']))


if __name__ == '__main__':
    main()
