import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils.random import sample_without_replacement
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression


class RandomSelection(BaseEstimator, TransformerMixin):
    """Random Selection of features"""
    def __init__(self, n_components=1000, random_state=None):
        self.n_components = n_components
        self.random_state = random_state
        self.components = None

    def fit(self, X, y=None):
        X = check_array(X)
        n_samples, n_features = X.shape

        random_state = check_random_state(self.random_state)
        self.components = sample_without_replacement(
                            n_features,
                            self.n_components,
                            random_state=random_state)
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["components"])
        X = check_array(X)
        n_samples, n_features = X.shape
        X_new = X[:, self.components]

        return X_new


class ZeroCutout(BaseEstimator, TransformerMixin):
    """Random Selection of features"""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = check_array(X)
        X = X.reshape(-1, 176, 208, 176)
        min_0 = float('inf')
        min_1 = float('inf')
        min_2 = float('inf')
        max_0 = float('-inf')
        max_1 = float('-inf')
        max_2 = float('-inf')
        for index in range(X.shape[0]):
            non_zero = np.where(X[index] > 0)
            min_0 = min(min_0, non_zero[0].min())
            min_1 = min(min_1, non_zero[1].min())
            min_2 = min(min_2, non_zero[2].min())
            max_0 = max(max_0, non_zero[0].max())
            max_1 = max(max_1, non_zero[1].max())
            max_2 = max(max_2, non_zero[2].max())

        X_new = X[:, min_0:max_0, min_1:max_1, min_2:max_2]
        print(X_new.shape)
        return X_new.reshape(X.shape[0], -1)


class SelectKBestRegression(SelectKBest):

    def __init__(self, k):
        super().__init__(score_func=f_regression, k=k)
