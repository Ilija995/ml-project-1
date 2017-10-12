import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


X = np.load('../../data/X_train.npy')
X = X.reshape(-1, 176, 208, 176)


def zero_cut(X):
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

def histogram(X, bandwidth):
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


X_new = zero_cut(X[:1, :, :, :])
X_histo = histogram(X_new, 450)
print(X_histo)




# z, x, y = X[0].nonzero()
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x, y, -z, zdir='z', c= 'red')
# plt.savefig("demo.png")

# new_shape = np.array(X.shape) / 2
# new_shape[0] = X.shape[0]
# new_data = np.zeros(shape=new_shape.astype(dtype=np.int32))

# print(new_shape)
# print(X.shape)


