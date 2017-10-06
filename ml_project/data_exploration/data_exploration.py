import numpy as np

train_data = np.load('../../data/X_train.npy')
# train_data = np.arange(10000100).reshape(100, 100001)
# train_data = np.random.rand(100, 100001)

print(train_data.shape)

m = train_data.mean(0, dtype=np.float64)
m = m[np.newaxis, :]
print(train_data.mean())

l = 6443008
# l = 100001
bucket_size = 2000
# bucket_size = 1000
o = int(round(l / bucket_size + 0.5))
counter = 0
new_data = np.zeros(shape=train_data.shape, dtype=np.float16)
for bucket in range(o):
    start = bucket * bucket_size
    end = min((bucket + 1) * bucket_size, l)
    data_bucket = train_data[:, start:end]
    bucket_mean = m[:, start:end]
    new_bucket = data_bucket - bucket_mean
    if counter < 10 and new_bucket.mean(dtype=np.float64) > 0.00001:
        counter += 1
        print('bucket {} shape {} start {} end {} mean {} new mean {}'.format(bucket, data_bucket.shape, start, end, data_bucket.mean(dtype=np.float64), new_bucket.mean(dtype=np.float64)))
    new_data[:, start:end] = new_bucket

print(new_data.shape)
print(new_data.mean(dtype=np.float64))
