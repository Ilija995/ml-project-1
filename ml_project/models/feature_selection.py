import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils.random import sample_without_replacement
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.neighbors import KernelDensity


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


class SelectFeatures(BaseEstimator, TransformerMixin):
    """Random Selection of features"""

    def __init__(self, bandwidth=100):
        self.bandwidth = bandwidth

    def fit(self, X, y=None):
        return self

    def pooling(self, X):
        X_new_shape = np.array(X.shape) // 2
        X_new_shape[0] = X.shape[0]
        X_new = np.zeros(shape=X_new_shape)
        d_x = [0, 1, 1, 0, 0, 1, 1, 0]
        d_y = [0, 0, 0, 0, 1, 1, 1, 1]
        d_z = [0, 0, 1, 1, 0, 0, 1, 1]
        for index in range(X_new.shape[0]):
            for x_axis in range(X_new.shape[1]):
                for y_axis in range(X_new.shape[2]):
                    for z_axis in range(X_new.shape[3]):
                        max_pixel = float('-inf')
                        for shift_index in range(len(d_x)):
                            max_pixel = max(max_pixel, X[index,
                                                         2 * x_axis + d_x[shift_index],
                                                         2 * y_axis + d_y[shift_index],
                                                         2 * z_axis + d_z[shift_index]])
                        X_new[index, x_axis, y_axis, z_axis] = max_pixel
        return X_new

    def zero_cut(self, X):
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

        min_0 -= (max_0 - min_0) % 2
        min_1 -= (max_1 - min_1) % 2
        min_2 -= (max_2 - min_2) % 2

        return X[:, min_0:max_0, min_1:max_1, min_2:max_2]

    def histogram(self, X, bandwidth):
        max_el = 4500
        no_buckets = max_el // bandwidth
        if max_el % bandwidth != 0:
            no_buckets += 1
        X_new = np.zeros(shape=[X.shape[0], X.shape[1] * no_buckets])
        for index in range(X.shape[0]):
            image = X[index]
            reshaped_image = image.reshape(image.shape[0], -1)
            hist = np.zeros(shape=[reshaped_image.shape[0], no_buckets])
            for i in range(reshaped_image.shape[0]):
                for j in range(reshaped_image.shape[1]):
                    hist[i][reshaped_image[i][j] // bandwidth] += 1
            X_new[index] = hist.reshape(-1)
        return X_new

    def transform(self, X, y=None):
        X = check_array(X)
        X = X.reshape(-1, 176, 208, 176)

        X_tmp = self.zero_cut(X)
        print("Temporary shape %s" % str(X_tmp.shape))

        # X_new = self.pooling(X_tmp)
        # print("New shape %s" % str(X_new.shape))
        # X_res = X_new.reshape(X_new.shape[0], -1)

        X_res = self.histogram(X_tmp, self.bandwidth)
        print("New shape %s" % str(X_res.shape))

        return X_res


class SelectKBestRegression(SelectKBest):

    def __init__(self, k):
        super().__init__(score_func=f_regression, k=k)
