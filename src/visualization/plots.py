import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy

def customized_scatterplot(y, x):
    ## Sizing the plot.
    style.use('fivethirtyeight')
    plt.subplots(figsize=(12, 8))
    ## Plotting target variable with predictor variable(OverallQual)
    sns.scatterplot(y=y, x=x)
    plt.show()


def linearity_regplot(y, x):
    fig, (ax1, ax2) = plt.subplots(figsize=(12, 8), ncols=2, sharey=False)
    sns.scatterplot(y, x, ax=ax1)
    sns.regplot(y, x, ax=ax1)
    plt.show()


def linearity_residplot(y, x):
    plt.subplots(figsize=(12, 8))
    sns.residplot(y, x)
    plt.show()


def all_features_heatmap(train_set):
    ## Plot fig sizing.
    style.use('ggplot')
    sns.set_style('whitegrid')
    plt.subplots(figsize=(30, 20))
    # Generate a mask for the upper triangle (taken from seaborn example gallery)
    mask = numpy.zeros_like(train_set.corr(), dtype=numpy.bool)
    mask[numpy.triu_indices_from(mask)] = True

    sns.heatmap(train_set.corr(),
                cmap=sns.diverging_palette(20, 220, n=200),
                mask=mask,
                annot=True,
                center=0,
                )
    ## Give title.
    plt.title("Heatmap of all the Features", fontsize=30)
    plt.show()

