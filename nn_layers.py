import numpy as np
from skimage.util.shape import view_as_windows


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    max = np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x - max)
    sum = np.sum(e_x, axis=-1, keepdims=True)
    f_x = e_x / sum
    return f_x


def convolve3d(x, W, b=0):
    y = view_as_windows(x, W.shape)

    y1 = y.reshape((y.shape[0], y.shape[1], y.shape[2], -1))
    y2 = y1.dot(W.reshape(-1)) + b

    return y2


class nn_convolutional_layer:

    def __init__(self, Wx_size, Wy_size, input_size, in_ch_size, out_ch_size, std=1e0):
    
        # initialization of weights
        self.W = np.random.normal(0, std / np.sqrt(in_ch_size * Wx_size * Wy_size / 2),
                                  (out_ch_size, in_ch_size, Wx_size, Wy_size))
        self.b = 0.01 + np.zeros((1, out_ch_size, 1, 1))
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
                padded = np.pad(spec_dLdy.reshape(1, spec_dLdy.shape[0], spec_dLdy.shape[1]),
                                ((pad_z, pad_z), (pad_y, pad_y), (pad_x, pad_x)), 'constant', constant_values=0)
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
                        dLdx_mat[col:col + plen, row:row + plen] += (mask[c_i, r_i] * dLdy_mat[c_i, r_i])

        return dLdx

class nn_fc_layer:

    def __init__(self, input_size, output_size, std=1):
        # Xavier/He init
        self.W = np.random.normal(0, std/np.sqrt(input_size/2), (output_size, input_size))
        self.b=0.01+np.zeros((output_size, 1))

    def forward(self, x):

        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)

        Wx = self.W @ x.T
        Wxb = Wx + self.b
        return Wxb.T

    def backprop(self, x, dLdy):

        batch_size = x.shape[0]
        x_full = x.reshape(batch_size, -1)

        dLdb = np.sum(dLdy, axis=0, keepdims=True)
        dLdx = np.dot(dLdy, self.W).reshape(x.shape)
        dLdW = np.dot(x_full.T, dLdy).T

        return dLdx, dLdW, dLdb

    def update_weights(self, dLdW, dLdb):

        # parameter update
        self.W=self.W+dLdW
        self.b=self.b+dLdb

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b


class nn_activation_layer:
    
    # performs ReLU activation
    def __init__(self):
        pass
    
    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask
    
    def backprop(self, x, dLdy):
        dLdx = self.mask * dLdy
        return dLdx


class nn_softmax_layer:

    def __init__(self):
        pass

    def forward(self, x):
        self.out = softmax(x)
        return self.out

    def backprop(self, x, dLdy):

        dLdx = np.zeros_like(dLdy)
        for ind in range(x.shape[0]):
            a = -np.outer(self.out[ind], self.out[ind])
            for j in range(x.shape[1]):
                a[j, j] = self.out[ind, j] * (1 - self.out[ind, j])
            dLdx[ind] = np.dot(dLdy[ind], a)

        return dLdx


class nn_cross_entropy_layer:

    def __init__(self):
        pass

    def forward(self, x, y):
        x = np.log(x)
        y = y.ravel()
        CEE = 0.0
        for ind in range(y.shape[0]):
            CEE += -x[ind, y[ind]]

        return CEE / x.shape[0]

    def backprop(self, x, y):
        dLdx = np.zeros_like(x)
        y = y.ravel()
        for ind in range(y.shape[0]):
            dLdx[ind, y[ind]] = -1 / x[ind, y[ind]]
        return dLdx
