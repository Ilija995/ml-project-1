import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array


class Normalize(BaseEstimator, TransformerMixin):
    """Normalize"""

    def __init__(self):
        self.bucket_size = 2000

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        min_X = X.min(0)
        min_X = min_X[np.newaxis, :]
        max_X = X.max(0)
        max_X = max_X[np.newaxis, :]

        no_features = X.shape[1]
        no_buckets = int(round(no_features / self.bucket_size + 0.5))
        for bucket_index in range(no_buckets):
            start = bucket_index * self.bucket_size
            end = min((bucket_index + 1) * self.bucket_size, no_features)
            data_bucket = X[:, start:end]
            min_bucket = min_X[:, start:end]
            max_bucket = max_X[:, start:end]
            diff_min_max = max_bucket - min_bucket
            diff_min_max[diff_min_max == 0] = 1
            X[:, start:end] = (data_bucket - min_bucket) / diff_min_max
        return X.reshape(X.shape[0], -1)
