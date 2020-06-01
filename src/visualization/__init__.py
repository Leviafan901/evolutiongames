import numpy
from src import load_drop_empty
from src.data_preparation.features_creation import TARGET_PREDICTION_VALUE
from src.data_processing.data_cleaning import removing_objects
from src.visualization.distribution import plotting_3_chart
from src.visualization.plots import customized_scatterplot, linearity_regplot, linearity_residplot


def main():
    data = load_drop_empty('../../data/train.csv', '../../data/test.csv')
    print(data.train_set.shape)
    print(data.test_set.shape)
    removing_objects(data)
    print((data.train_set.corr() ** 2)[TARGET_PREDICTION_VALUE].sort_values(ascending=False)[1:])
    #plotting_3_chart(data.train_set, TARGET_PREDICTION_VALUE)
    #customized_scatterplot(data.train_set.charges, data.train_set.smoker)
    #customized_scatterplot(data.train_set.charges, data.train_set.age)
    #linearity(data.train_set.smoker, data.train_set.charges)
    #linearity_regplot(data.train_set.age, data.train_set.charges)
    #linearity_residplot(data.train_set.age, data.train_set.charges)
    previous_data = data
    data.train_set['charges'] = numpy.log1p(data.train_set['charges'])
    linearity_residplot(previous_data.train_set.age, previous_data.train_set.charges)
    linearity_residplot(data.train_set.age, data.train_set.charges)


if __name__ == '__main__':
    main()
