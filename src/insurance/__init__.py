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

    #Replacing str values with boolean 0 and 1 values
    sex = {'male': 0, 'female': 1}
    train.sex = [sex[item] for item in train.sex]
    test.sex = [sex[item] for item in test.sex]
    smoker = {'no': 0, 'yes': 1}
    train.smoker = [smoker[item] for item in train.smoker]
    test.smoker = [smoker[item] for item in test.smoker]
    region = {'northwest': 1, 'southeast': 2, 'northeast': 3, 'southwest': 4}
    train.region = [region[item] for item in train.region]
    test.region = [region[item] for item in test.region]

   # X_train = np.array(train.iloc[:, :-1].values)
   # y_train = np.array(train.iloc[:, 1].values)

   # model = LinearRegression()
    #model.fit(X_train, y_train)
    #plt.plot(X_train, model.predict(X_train), color='green')
   # plt.show()

    def plotting_3_chart(df, feature):
        ## Importing seaborn, matplotlab and scipy modules.
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from scipy import stats
        import matplotlib.style as style
        style.use('fivethirtyeight')

        ## Creating a customized chart. and giving in figsize and everything.
        fig = plt.figure(constrained_layout=True, figsize=(12, 8))
        ## creating a grid of 3 cols and 3 rows.
        grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
        # gs = fig3.add_gridspec(3, 3)

        ## Customizing the histogram grid.
        ax1 = fig.add_subplot(grid[0, :2])
        ## Set the title.
        ax1.set_title('Histogram')
        ## plot the histogram.
        sns.distplot(df.loc[:, feature], norm_hist=True, ax=ax1)

        # customizing the QQ_plot.
        ax2 = fig.add_subplot(grid[1, :2])
        ## Set the title.
        ax2.set_title('QQ_plot')
        ## Plotting the QQ_Plot.
        stats.probplot(df.loc[:, feature], plot=ax2)

        ## Customizing the Box Plot.
        ax3 = fig.add_subplot(grid[:, 2])
        ## Set title.
        ax3.set_title('Box Plot')
        ## Plotting the box plot.
        sns.boxplot(df.loc[:, feature], orient='v', ax=ax3)
        plt.show()

    #plotting_3_chart(train, 'charges')
    print("Skewness: " + str(train['charges'].skew()))
    print("Kurtosis: " + str(train['charges'].kurt()))

    print((train.corr() ** 2)["charges"].sort_values(ascending=False)[1:])

    #normal distribution
    train["charges"] = np.log1p(train["charges"])
    #plotting_3_chart(train, 'charges')

    sns.distplot(train['smoker'])
    import data_cleaning as dc
    dc.fixing_skewness(train)
    dc.fixing_skewness(test)
    print("Skewness: " + str(train['charges'].skew()))
    final_features = pd.get_dummies(train).reset_index(drop=True)
    final_features = pd.get_dummies(test).reset_index(drop=True)
    print(final_features.shape)