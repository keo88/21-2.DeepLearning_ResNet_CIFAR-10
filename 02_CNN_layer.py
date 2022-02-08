import numpy as np
from skimage.util.shape import view_as_windows
import time


def convolve3d(x, W, b=0):
    y = view_as_windows(x, W.shape)

    y1 = y.reshape((y.shape[0], y.shape[1], y.shape[2], -1))
    y2 = y1.dot(W.reshape(-1)) + b

    return y2


class nn_convolutional_layer:

    def __init__(self, filter_width, filter_height, input_size, in_ch_size, num_filters, std=1e0):
        # initialization of weights
        self.W = np.random.normal(0, std / np.sqrt(in_ch_size * filter_width * filter_height / 2),
                                  (num_filters, in_ch_size, filter_width, filter_height))
        self.b = 0.01 + np.zeros((1, num_filters, 1, 1))
        self.input_size = input_size

    def update_weights(self, dW, db):
        self.W += dW
        self.b += db

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b

    def forward(self, x):

        batch_size = x.shape[0]
        num_filter = self.W.shape[0]
        height = x.shape[2] - self.W.shape[2] + 1
        width = x.shape[3] - self.W.shape[3] + 1

        out = np.zeros((batch_size, num_filter, height, width))

        for batch in range(batch_size):
            for filter in range(num_filter):
                out[batch][filter] = convolve3d(x[batch], self.W[filter], self.b[0, filter, 0, 0])[0]

        return out

    def backprop(self, x, dLdy):
        batch_size = dLdy.shape[0]
        num_filter = dLdy.shape[1]

        pad_x = self.W.shape[-2] - 1
        pad_y = self.W.shape[-1] - 1
        pad_z = self.W.shape[-3] - 1

        dLdx = np.zeros_like(x)
        dLdW = np.zeros_like(self.W)
        dLdb = np.zeros_like(self.b)

        for batch in range(batch_size):

            reversed_x = np.flip(x[batch])
            for filter in range(num_filter):
                spec_dLdy = dLdy[batch][filter]
                padded = np.pad(spec_dLdy.reshape(1, spec_dLdy.shape[0], spec_dLdy.shape[1]), ((pad_z, pad_z), (pad_y, pad_y), (pad_x, pad_x)), 'constant', constant_values=0)
                reversed_W = np.flip(self.W[filter])

                dLdx[batch] += convolve3d(padded, reversed_W)
                dLdW[filter] += convolve3d(padded, reversed_x)
                dLdb[0, filter, 0, 0] += np.sum(spec_dLdy)
        return dLdx, dLdW, dLdb


class nn_max_pooling_layer:
    def __init__(self, stride, pool_size):
        self.stride = stride
        self.pool_size = pool_size

    def forward(self, x):
        batch_size = x.shape[0]
        channel_size = x.shape[1]
        height = x.shape[2]
        width = x.shape[3]

        pool_height = (height - self.pool_size) // self.stride + 1
        pool_width = (width - self.pool_size) // self.stride + 1

        out = np.zeros((batch_size, channel_size, pool_height, pool_width))
        self.max_mask = np.zeros((batch_size, channel_size, pool_height, pool_width, self.pool_size, self.pool_size))

        for batch in range(batch_size):
            for channel in range(channel_size):
                x_mat = x[batch][channel]
                y = view_as_windows(x_mat, (self.pool_size, self.pool_size), step=self.stride)
                y1 = np.max(y, axis=(-2, -1), keepdims=True)
                out[batch][channel] = y1.reshape(pool_height, pool_width)
                y2 = (y1 == y).astype(int)
                self.max_mask[batch][channel] = y2
        return out

    def backprop(self, x, dLdy):

        plen = self.pool_size
        batch_size = x.shape[0]
        channel_size = x.shape[1]
        height = x.shape[2]
        width = x.shape[3]

        dLdx = np.zeros_like(x, dtype='float64')

        for batch in range(batch_size):
            for channel in range(channel_size):
                dLdx_mat = dLdx[batch, channel]
                dLdy_mat = dLdy[batch, channel]
                mask = self.max_mask[batch, channel]

                for c_i, col in enumerate(range(0, height - 1, self.pool_size)):
                    for r_i, row in enumerate(range(0, width - 1, self.pool_size)):
                        dLdx_mat[col:col+plen, row:row+plen] += (mask[c_i, r_i] * dLdy_mat[c_i, r_i])
        return dLdx


start = time.time()

# data sizes
batch_size = 8
input_size = 32
filter_width = 3
filter_height = filter_width
in_ch_size = 3
num_filters = 8

std = 1e0
dt = 1e-3

# number of test loops
num_test = 20

# error parameters
err_dLdb = 0
err_dLdx = 0
err_dLdW = 0
err_dLdx_pool = 0

for i in range(num_test):
    # create convolutional layer object
    cnv = nn_convolutional_layer(filter_width, filter_height, input_size, in_ch_size, num_filters, std)

    x = np.random.normal(0, 1, (batch_size, in_ch_size, input_size, input_size))
    delta = np.random.normal(0, 1, (batch_size, in_ch_size, input_size, input_size)) * dt

    # dLdx test
    print('dLdx test')
    y1 = cnv.forward(x)
    y2 = cnv.forward(x + delta)

    bp, _, _ = cnv.backprop(x, np.ones(y1.shape))

    exact_dx = np.sum(y2 - y1) / dt
    apprx_dx = np.sum(delta * bp) / dt
    print('exact change', exact_dx)
    print('apprx change', apprx_dx)

    err_dLdx += abs((apprx_dx - exact_dx) / exact_dx) / num_test * 100

    # dLdW test
    print('dLdW test')
    W, b = cnv.get_weights()
    dW = np.random.normal(0, 1, W.shape) * dt
    db = np.zeros(b.shape)

    z1 = cnv.forward(x)
    _, bpw, _ = cnv.backprop(x, np.ones(z1.shape))
    cnv.update_weights(dW, db)
    z2 = cnv.forward(x)

    exact_dW = np.sum(z2 - z1) / dt
    apprx_dW = np.sum(dW * bpw) / dt
    print('exact change', exact_dW)
    print('apprx change', apprx_dW)

    err_dLdW += abs((apprx_dW - exact_dW) / exact_dW) / num_test * 100

    # dLdb test
    print('dLdb test')

    W, b = cnv.get_weights()

    dW = np.zeros(W.shape)
    db = np.random.normal(0, 1, b.shape) * dt

    z1 = cnv.forward(x)

    V = np.random.normal(0, 1, z1.shape)

    _, _, bpb = cnv.backprop(x, V)

    cnv.update_weights(dW, db)
    z2 = cnv.forward(x)

    exact_db = np.sum(V * (z2 - z1) / dt)
    apprx_db = np.sum(db * bpb) / dt

    print('exact change', exact_db)
    print('apprx change', apprx_db)
    err_dLdb += abs((apprx_db - exact_db) / exact_db) / num_test * 100

    # max pooling test
    # parameters for max pooling
    stride = 2
    pool_size = 2

    mpl = nn_max_pooling_layer(stride=stride, pool_size=pool_size)

    x = np.arange(batch_size * in_ch_size * input_size * input_size).reshape(
        (batch_size, in_ch_size, input_size, input_size)) + 1
    delta = np.random.normal(0, 1, (batch_size, in_ch_size, input_size, input_size)) * dt

    print('dLdx test for pooling')
    y1 = mpl.forward(x)
    dLdy = np.random.normal(0, 10, y1.shape)
    bpm = mpl.backprop(x, dLdy)

    y2 = mpl.forward(x + delta)

    exact_dx_pool = np.sum(dLdy * (y2 - y1)) / dt
    apprx_dx_pool = np.sum(delta * bpm) / dt
    print('exact change', exact_dx_pool)
    print('apprx change', apprx_dx_pool)

    err_dLdx_pool += abs((apprx_dx_pool - exact_dx_pool) / exact_dx_pool) / num_test * 100

# reporting accuracy results.
print('accuracy results')
print('conv layer dLdx', 100 - err_dLdx, '%')
print('conv layer dLdW', 100 - err_dLdW, '%')
print('conv layer dLdb', 100 - err_dLdb, '%')
print('maxpool layer dLdx', 100 - err_dLdx_pool, '%')

print('Total Elapsed Time: ' + str(time.time() - start))