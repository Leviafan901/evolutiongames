from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor


def linear_regression(X_train, X_test, y_train, y_test):
    lin_reg_model = LinearRegression(normalize=True, n_jobs=-1)
    lin_reg_model.fit(X_train, y_train)
    lin_reg_model.predict(X_test)
    accuracy = lin_reg_model.score(X_test, y_test)
    print(f'Linear regression accuracy = %.2f' % accuracy)
