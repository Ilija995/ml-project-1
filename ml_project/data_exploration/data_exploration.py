import numpy as np

X = np.load('../../data/X_train.npy')

min_X = X.min(0)
min_X = min_X[np.newaxis, :]
max_X = X.max(0)
max_X = max_X[np.newaxis, :]

no_features = 6443008
bucket_size = 2000
no_buckets = int(round(no_features / bucket_size + 0.5))
for bucket_index in range(no_buckets):
    start = bucket_index * bucket_size
    end = min((bucket_index + 1) * bucket_size, no_features)
    data_bucket = X[:, start:end]
    min_bucket = min_X[:, start:end]
    max_bucket = max_X[:, start:end]
    diff_min_max = max_bucket - min_bucket
    diff_min_max[diff_min_max == 0] = 1
    X[:, start:end] = (data_bucket - min_bucket) / diff_min_max

print(X.min().mean())
print(X.max().mean())
