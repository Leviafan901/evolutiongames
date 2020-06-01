import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style

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
