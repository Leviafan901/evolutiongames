from src import load_drop_empty
from src.data_preparation.features_creation import TARGET_PREDICTION_VALUE
from src.visualization.distribution import plotting_3_chart


def main():
    data = load_drop_empty('../../data/train.csv', '../../data/test.csv')
    plotting_3_chart(data.train_set, TARGET_PREDICTION_VALUE)


if __name__ == '__main__':
    main()
