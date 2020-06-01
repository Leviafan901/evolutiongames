from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax


def fixing_skewness(data):
    ## Getting all the data that are not of "object" type.
    numeric_feats = data.dtypes[data.dtypes != "object"].index
    # Check the skew of all numerical features
    skewed_feats = data[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)
    high_skew = skewed_feats[abs(skewed_feats) > 0.5]
    skewed_features = high_skew.index
    for feat in skewed_features:
        data[feat] = boxcox1p(data[feat], boxcox_normmax(data[feat] + 1))


def removing_objects(data):
    sets = (data.train_set, data.test_set)
    for set in sets:
        # Replacing str values with boolean 0 and 1 values
        sex = {'male': 0, 'female': 1}
        set.sex = [sex[item] for item in set.sex]
        smoker = {'no': 0, 'yes': 1}
        set.smoker = [smoker[item] for item in set.smoker]
        region = {'northwest': 1, 'southeast': 2, 'northeast': 3, 'southwest': 4}
        set.region = [region[item] for item in set.region]
