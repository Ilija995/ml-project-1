import math
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class HistogramDownsampling(BaseEstimator, TransformerMixin):
    """Downsampling data with histogram"""
    def __init__(self, bins=10, kernel_size=(16, 16)):
        self.bins = bins
        self.kernel_size = kernel_size

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_reshaped = X.reshape(-1, 176, 208, 176)
        X_cropped = X_reshaped[:, 40:140, 40:170, 40:140]
        no_histograms = math.ceil(X_cropped.shape[2] / self.kernel_size[0]) * math.ceil(X_cropped.shape[3] / self.kernel_size[1])
        X_hist = np.zeros(shape=[X_cropped.shape[0], X_cropped.shape[1], no_histograms, self.bins])
        for sample_index in range(X_cropped.shape[0]):
            print((sample_index + 1) / X_cropped.shape[0] * 100)
            for layer_index in range(X_cropped.shape[1]):
                layer = X_cropped[sample_index][layer_index]
                no_j_buckets = math.ceil(X_cropped.shape[3] / self.kernel_size[1])
                for bucket_i in range(0, layer.shape[0], self.kernel_size[0]):
                    for bucket_j in range(0, layer.shape[1], self.kernel_size[1]):
                        bucket = layer[bucket_i:min(bucket_i + self.kernel_size[0], layer.shape[0]),
                                 bucket_j:min(bucket_j + self.kernel_size[1], layer.shape[1])]
                        hist, bin_edges = np.histogram(bucket.reshape(-1), bins=self.bins)
                        hist_index = (bucket_i // self.kernel_size[0]) * no_j_buckets + (bucket_j // self.kernel_size[1])
                        X_hist[sample_index][layer_index][hist_index] = hist
        return X_hist.reshape(X_hist.shape[0], -1)
