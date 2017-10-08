import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


X = np.load('../../data/X_train.npy')
X = X.reshape(-1, 176, 208, 176)

print(np.count_nonzero(X == 0) / X.size)
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

X_new = X[-1, min_0:max_0, min_1:max_1, min_2:max_2]

print(min_0)
print(min_1)
print(min_2)
print(max_0)
print(max_1)
print(max_2)
print(X_new.shape)



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


