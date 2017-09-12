import struct

import numpy as np

import scipy as sp

import time

def read_idx(filename, path='./mldata/'):
    with open(path + filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

    # Create an iterator which returns each image in turn
    # for i in xrange(len(lbl)):
    #     yield get_img(i)

    return img, lbl

def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()


def one_hot(labels):
    classes = np.unique(labels).astype(int)
    n_classes = classes.size
    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in classes:
        one_hot_labels[labels == c, c] = 1
    return one_hot_labels


def unhot(one_hot_labels):
    return np.argmax(one_hot_labels, axis=-1)


def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))


def sigmoid_d(x):
    s = sigmoid(x)
    return s*(1-s)


def tanh(x):
    return np.tanh(x)


def tanh_d(x):
    e = np.exp(2*x)
    return (e-1)/(e+1)


def relu(x):
    return np.maximum(0.0, x)


def relu_d(x):
    dx = np.zeros(x.shape)
    dx[x >= 0] = 1
    return dx

class Layer(object):
    def _setup(self, input_shape, rng):
        """ Setup layer with parameters that are unknown at __init__(). """
        pass

    def fprop(self, input):
        """ Calculate layer output for given input (forward propagation). """
        raise NotImplementedError()

    def bprop(self, output_grad):
        """ Calculate input gradient. """
        raise NotImplementedError()

    def output_shape(self, input_shape):
        """ Calculate shape of this layer's output.
        input_shape[0] is the number of samples in the input.
        input_shape[1:] is the shape of the feature.
        """
        raise NotImplementedError()


class LossMixin(object):
    def loss(self, output, output_pred):
        """ Calculate mean loss given output and predicted output. """
        raise NotImplementedError()

    def input_grad(self, output, output_pred):
        """ Calculate input gradient given output and predicted output. """
        raise NotImplementedError()


class ParamMixin(object):
    def params(self):
        """ Layer parameters. """
        raise NotImplementedError()

    def param_grads(self):
        """ Get layer parameter gradients as calculated from bprop(). """
        raise NotImplementedError()

    def param_incs(self):
        """ Get layer parameter steps as calculated from bprop(). """
        raise NotImplementedError()


class Linear(Layer, ParamMixin):
    def __init__(self, n_out, weight_scale, weight_decay=0.0):
        self.n_out = n_out
        self.weight_scale = weight_scale
        self.weight_decay = weight_decay

    def _setup(self, input_shape, rng):
        n_input = input_shape[1]
        W_shape = (n_input, self.n_out)
        self.W = rng.normal(size=W_shape, scale=self.weight_scale)
        self.b = np.zeros(self.n_out)

    def fprop(self, input):
        self.last_input = input
        return np.dot(input, self.W) + self.b

    def bprop(self, output_grad):
        n = output_grad.shape[0]
        self.dW = np.dot(self.last_input.T, output_grad)/n - self.weight_decay*self.W
        self.db = np.mean(output_grad, axis=0)
        return np.dot(output_grad, self.W.T)

    def params(self):
        return self.W, self.b

    def param_incs(self):
        return self.dW, self.db

    def param_grads(self):
        # undo weight decay to get gradient
        gW = self.dW+self.weight_decay*self.W
        return gW, self.db

    def output_shape(self, input_shape):
        return (input_shape[0], self.n_out)


class Activation(Layer):
    def __init__(self, type):
        if type == 'sigmoid':
            self.fun = sigmoid
            self.fun_d = sigmoid_d
        elif type == 'relu':
            self.fun = relu
            self.fun_d = relu_d
        elif type == 'tanh':
            self.fun = tanh
            self.fun_d = tanh_d
        else:
            raise ValueError('Invalid activation function.')

    def fprop(self, input):
        self.last_input = input
        return self.fun(input)

    def bprop(self, output_grad):
        return output_grad*self.fun_d(self.last_input)

    def output_shape(self, input_shape):
        return input_shape


class LogRegression(Layer, LossMixin):
    """ Softmax layer with cross-entropy loss function. """
    def fprop(self, input):
        e = np.exp(input - np.amax(input, axis=1, keepdims=True))
        return e/np.sum(e, axis=1, keepdims=True)

    def bprop(self, output_grad):
        raise NotImplementedError(
            'LogRegression does not support back-propagation of gradients. '
            + 'It should occur only as the last layer of a NeuralNetwork.'
        )

    def input_grad(self, Y, Y_pred):
        # Assumes one-hot encoding.
        return -(Y - Y_pred)

    def loss(self, Y, Y_pred):
        # Assumes one-hot encoding.
        eps = 1e-15
        Y_pred = np.clip(Y_pred, eps, 1 - eps)
        Y_pred /= Y_pred.sum(axis=1, keepdims=True)
        loss = -np.sum(Y * np.log(Y_pred))
        return loss / Y.shape[0]

    def output_shape(self, input_shape):
        return input_shape


class NeuralNetwork:
    def __init__(self, layers, rng=None):
        self.layers = layers
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng

    def _setup(self, X, Y):
        # Setup layers sequentially
        next_shape = X.shape
        for layer in self.layers:
            layer._setup(next_shape, self.rng)
            next_shape = layer.output_shape(next_shape)
#            print(next_shape)
        if next_shape != Y.shape:
            raise ValueError('Output shape %s does not match Y %s'
                             % (next_shape, Y.shape))

    def fit(self, X, Y, learning_rate=0.1, max_iter=10, batch_size=64):
        """ Train network on the given data. """
        n_samples = Y.shape[0]
        n_batches = n_samples // batch_size
        Y_one_hot = one_hot(Y)
        self._setup(X, Y_one_hot)
        iter = 0
        # Stochastic gradient descent with mini-batches
        while iter < max_iter:
            iter += 1
            for b in range(n_batches):
                batch_start = time.time()
                batch_begin = b*batch_size
                batch_end = batch_begin+batch_size
                X_batch = X[batch_begin:batch_end]
                Y_batch = Y_one_hot[batch_begin:batch_end]

                # Forward propagation
                X_next = X_batch
                for layer in self.layers:
                    X_next = layer.fprop(X_next)
                Y_pred = X_next

                # Back propagation of partial derivatives
                next_grad = self.layers[-1].input_grad(Y_batch, Y_pred)
                for layer in reversed(self.layers[:-1]):
                    next_grad = layer.bprop(next_grad)

                # Update parameters
                for layer in self.layers:
                    if isinstance(layer, ParamMixin):
                        for param, inc in zip(layer.params(),
                                              layer.param_incs()):
                            param -= learning_rate*inc
                batch_end = time.time()
                print("Running batch", b, "through the network took", round(batch_end-batch_start, 3), "seconds")
            # Output training status
            print("Computing loss")
            loss_start = time.time()
            loss = self._loss(X, Y_one_hot)
            print("Loss", loss)
            loss_end = time.time()
            print("Computing loss", loss, "took", round(loss_end-loss_start, 3), "seconds")
            print("Computing error")
            error_start = time.time()
            error = self.error(X, Y)
            error_end = time.time()
            print("Error", error)
            print("Computing error", error, "took", round(error_end-error_start, 3), "seconds")
            print('iter %i, loss %.4f, train error %.4f' % (iter, loss, error))

    def _loss(self, X, Y_one_hot):
        X_next = X
        for layer in self.layers:
            X_next = layer.fprop(X_next)
        Y_pred = X_next
        return self.layers[-1].loss(Y_one_hot, Y_pred)

    def predict(self, X):
        """ Calculate an output Y for the given input X. """
        X_next = X
        for layer in self.layers:
            X_next = layer.fprop(X_next)
        Y_pred = unhot(X_next)
        return Y_pred

    def error(self, X, Y):
        """ Calculate error on the given data. """
        Y_pred = self.predict(X)
        error = Y_pred != Y
        return np.mean(error)

    def check_gradients(self, X, Y):
        """ Helper function to test the parameter gradients for
        correctness. """
        # Warning: the following is a hack
        Y_one_hot = one_hot(Y)
        self._setup(X, Y_one_hot)
        for l, layer in enumerate(self.layers):
            if isinstance(layer, ParamMixin):
                print('layer %d' % l)
                for p, param in enumerate(layer.params()):
                    param_shape = param.shape

                    def fun(param_new):
                        param[:] = np.reshape(param_new, param_shape)
                        return self._loss(X, Y_one_hot)

                    def grad_fun(param_new):
                        param[:] = np.reshape(param_new, param_shape)
                        # Forward propagation
                        X_next = X
                        for layer in self.layers:
                            X_next = layer.fprop(X_next)
                        Y_pred = X_next

                        # Back-propagation of partial derivatives
                        next_grad = self.layers[-1].input_grad(Y_one_hot,
                                                               Y_pred)
                        for layer in reversed(self.layers[l:-1]):
                            next_grad = layer.bprop(next_grad)
                        return np.ravel(self.layers[l].param_grads()[p])

                    param_init = np.ravel(np.copy(param))
                    err = sp.optimize.check_grad(fun, grad_fun, param_init)
                    print('diff %.2e' % err)


class Conv(Layer, ParamMixin):
    def __init__(self, n_feats, filter_shape, strides, weight_scale,
                 weight_decay=0.0, padding_mode='same', border_mode='nearest'):
        self.n_feats = n_feats
        self.filter_shape = filter_shape
        self.strides = strides
        self.weight_scale = weight_scale
        self.weight_decay = weight_decay
        self.padding_mode = padding_mode
        self.border_mode = border_mode

    def _setup(self, input_shape, rng):
        n_channels = input_shape[1]
        W_shape = (n_channels, self.n_feats) + self.filter_shape
        self.W = rng.normal(size=W_shape, scale=self.weight_scale)
        self.b = np.zeros(self.n_feats)

    def fprop(self, input):
        self.last_input = input
        self.last_input_shape = input.shape
        convout = np.empty(self.output_shape(input.shape))
        convout = conv_bc01(input, self.W, convout)
        return convout + self.b[np.newaxis, :, np.newaxis, np.newaxis]

    def bprop(self, output_grad):
        input_grad = np.empty(self.last_input_shape)
        self.dW = np.empty(self.W.shape)
        input_grad, self.dW = bprop_conv_bc01(self.last_input, output_grad,
                                              self.W, input_grad, self.dW)
        n_imgs = output_grad.shape[0]
        self.db = np.sum(output_grad, axis=(0, 2, 3)) / (n_imgs)
        self.dW -= self.weight_decay*self.W
        return input_grad

    def params(self):
        return self.W, self.b

    def param_incs(self):
        return self.dW, self.db

    def param_grads(self):
        # undo weight decay
        gW = self.dW+self.weight_decay*self.W
        return gW, self.db

    def output_shape(self, input_shape):
        if self.padding_mode == 'same':
            h = input_shape[2]
            w = input_shape[3]
        elif self.padding_mode == 'full':
            h = input_shape[2]-self.filter_shape[1]+1
            w = input_shape[3]-self.filter_shape[2]+1
        else:
            h = input_shape[2]+self.filter_shape[1]-1
            w = input_shape[3]+self.filter_shape[2]-1
        shape = (input_shape[0], self.n_feats, h, w)
        return shape


class Pool(Layer):
    def __init__(self, pool_shape=(3, 3), strides=(1, 1), mode='max'):
        self.mode = mode
        self.pool_h, self.pool_w = pool_shape
        self.stride_y, self.stride_x = strides

    def fprop(self, input):
        self.last_input_shape = input.shape
        self.last_switches = np.empty(self.output_shape(input.shape)+(2,),
                                      dtype=np.int)
        poolout = np.empty(self.output_shape(input.shape))
        poolout = pool_bc01(input, poolout, self.last_switches, self.pool_h, self.pool_w,
                  self.stride_y, self.stride_x)
        return poolout

    def bprop(self, output_grad):
        input_grad = np.empty(self.last_input_shape)
        input_grad = bprop_pool_bc01(output_grad, self.last_switches, input_grad)
        return input_grad

    def output_shape(self, input_shape):
        shape = (input_shape[0],
                 input_shape[1],
                 input_shape[2]//self.stride_y,
                 input_shape[3]//self.stride_x)
        return shape


class Flatten(Layer):
    def fprop(self, input):
        self.last_input_shape = input.shape
        return np.reshape(input, (input.shape[0], -1))

    def bprop(self, output_grad):
        return np.reshape(output_grad, self.last_input_shape)

    def output_shape(self, input_shape):
        return (input_shape[0], np.prod(input_shape[1:]))

def conv_bc01(imgs, filters, convout):
    """ Multi-image, multi-channel convolution
    imgs has shape (n_imgs, n_channels_in, img_h, img_w)
    filters has shape (n_channels_in, n_channels_out, img_h, img_w)
    convout has shape (n_imgs, n_channels_out, img_h, img_w)
    """
    # TODO: support padding and striding
    # TODO: experiment with border mode 'reflect'

    n_imgs = imgs.shape[0]
    img_h = imgs.shape[2]
    img_w = imgs.shape[3]
    n_channels_in = filters.shape[0]
    n_channels_out = filters.shape[1]
    fil_h = filters.shape[2]
    fil_w = filters.shape[3]

    fil_mid_h = fil_h // 2
    fil_mid_w = fil_w // 2

    for i in range(n_imgs):
        for c_out in range(n_channels_out):
            for y in range(img_h):
                y_off_min = max(-y, -fil_mid_h)
                y_off_max = min(img_h-y, fil_mid_h+1)
                for x in range(img_w):
                    x_off_min = max(-x, -fil_mid_w)
                    x_off_max = min(img_w-x, fil_mid_w+1)
                    value = 0.0
                    for y_off in range(y_off_min, y_off_max):
                        for x_off in range(x_off_min, x_off_max):
                            img_y = y + y_off
                            img_x = x + x_off
                            fil_y = fil_mid_w + y_off
                            fil_x = fil_mid_h + x_off
                            for c_in in range(n_channels_in):
                                value += imgs[i, c_in, img_y, img_x] * filters[c_in, c_out, fil_y, fil_x]
                    convout[i, c_out, y, x] = value

    return convout

def bprop_conv_bc01(imgs, convout_grad, filters, imgs_grad, filters_grad):
    """ Back-propagate gradients of multi-image, multi-channel convolution
    imgs has shape (n_imgs, n_channels_in, img_h, img_w)
    filters has shape (n_channels_in, n_channels_out, img_h, img_w)
    convout has shape (n_imgs, n_channels_out, img_h, img_w)
    """

    n_imgs = convout_grad.shape[0]
    img_h = convout_grad.shape[2]
    img_w = convout_grad.shape[3]
    n_channels_convout = filters.shape[1]
    n_channels_imgs = filters.shape[0]
    fil_h = filters.shape[2]
    fil_w = filters.shape[3]
    fil_mid_h = fil_h // 2
    fil_mid_w = fil_w // 2

    imgs_grad = np.zeros((n_imgs, n_channels_imgs, img_h, img_h))
    filters_grad = np.zeros((n_channels_imgs, n_channels_convout, fil_h, fil_w))
    for i in range(n_imgs):
        for c_convout in range(n_channels_convout):
            for y in range(img_h):
                y_off_min = max(-y, -fil_mid_h)
                y_off_max = min(img_h-y, fil_mid_h+1)
                for x in range(img_w):
                    convout_grad_value = convout_grad[i, c_convout, y, x]
                    x_off_min = max(-x, -fil_mid_w)
                    x_off_max = min(img_w-x, fil_mid_w+1)
                    for y_off in range(y_off_min, y_off_max):
                        for x_off in range(x_off_min, x_off_max):
                            img_y = y + y_off
                            img_x = x + x_off
                            fil_y = fil_mid_w + y_off
                            fil_x = fil_mid_h + x_off
                            for c_imgs in range(n_channels_imgs):
                                imgs_grad[i, c_imgs, img_y, img_x] += filters[c_imgs, c_convout, fil_y, fil_x] * convout_grad_value
                                filters_grad[c_imgs, c_convout, fil_y, fil_x] += imgs[i, c_imgs, img_y, img_x] * convout_grad_value
    filters_grad /= n_imgs

    return imgs_grad, filters_grad

def pool_bc01(imgs, poolout, switches, pool_h,
              pool_w, stride_y, stride_x):
    """ Multi-image, multi-channel pooling
    imgs has shape (n_imgs, n_channels, img_h, img_w)
    poolout has shape (n_imgs, n_channels, img_h//stride_y, img_w//stride_x)
    switches has shape (n_imgs, n_channels, img_h//stride_y, img_w//stride_x, 2)
    """
    # TODO: mean pool

    n_imgs = imgs.shape[0]
    n_channels = imgs.shape[1]
    img_h = imgs.shape[2]
    img_w = imgs.shape[3]

    out_h = img_h // stride_y
    out_w = img_w // stride_x

    pool_h_top = pool_h // 2 - 1 + pool_h % 2
    pool_h_bottom = pool_h // 2 + 1
    pool_w_left = pool_w // 2 - 1 + pool_w % 2
    pool_w_right = pool_w // 2 + 1

    if not n_imgs == poolout.shape[0] == switches.shape[0]:
        raise ValueError('Mismatch in number of images.')
    if not n_channels == poolout.shape[1] == switches.shape[1]:
        raise ValueError('Mismatch in number of channels.')
    if not (out_h == poolout.shape[2] == switches.shape[2] and out_w == poolout.shape[3] == switches.shape[3]):
        raise ValueError('Mismatch in image shape.')
    if not switches.shape[4] == 2:
        raise ValueError('switches should only have length 2 in the 5. dimension.')

    img_y_max = 0
    img_x_max = 0

    poolout = np.zeros((n_imgs, n_channels, out_h, out_w))
    for i in range(n_imgs):
        for c in range(n_channels):
            for y_out in range(out_h):
                y = y_out*stride_y
                y_min = max(y-pool_h_top, 0)
                y_max = min(y+pool_h_bottom, img_h)
                for x_out in range(out_w):
                    x = x_out*stride_x
                    x_min = max(x-pool_w_left, 0)
                    x_max = min(x+pool_w_right, img_w)
                    value = -9e99
                    for img_y in range(y_min, y_max):
                        for img_x in range(x_min, x_max):
                            new_value = imgs[i, c, img_y, img_x]
                            if new_value > value:
                                value = new_value
                                img_y_max = img_y
                                img_x_max = img_x
                    poolout[i, c, y_out, x_out] = value
                    switches[i, c, y_out, x_out, 0] = img_y_max
                    switches[i, c, y_out, x_out, 1] = img_x_max

    return poolout

def bprop_pool_bc01(poolout_grad, switches, imgs_grad):

    n_imgs = poolout_grad.shape[0]
    n_channels = poolout_grad.shape[1]
    poolout_h = poolout_grad.shape[2]
    poolout_w = poolout_grad.shape[3]

    imgs_grad = np.zeros((n_imgs, n_channels, imgs_grad.shape[2], imgs_grad.shape[3]))
    for i in range(n_imgs):
        for c in range(n_channels):
            for y in range(poolout_h):
                for x in range(poolout_w):
                    img_y = switches[i, c, y, x, 0]
                    img_x = switches[i, c, y, x, 1]
                    imgs_grad[i, c, img_y, img_x] = poolout_grad[i, c, y, x]
    return imgs_grad

if __name__=="__main__":

    mnist_data = read_idx('train-images-idx3-ubyte')
    mnist_target = read_idx('train-labels-idx1-ubyte')

    split = 48000

    ## TODO: read in MNIST data from mnist-original.mat into numpy arrays
    ## TODO: Change code to read from original MNIST files, following here:
    ## https://gist.github.com/akesling/5358964
    X_train = np.reshape(mnist_data[:split], (-1, 1, 28, 28))/255.0
    y_train = mnist_target[:split]
    X_test = np.reshape(mnist_data[split:], (-1, 1, 28, 28))/255.0
    y_test = mnist_target[split:]
    n_classes = np.unique(y_train).size

    # Downsample training data
    n_train_samples = 3000
    train_idxs = np.random.randint(0, split-1, n_train_samples)
    X_train = X_train[train_idxs, ...]
    y_train = y_train[train_idxs, ...]

    # Setup convolutional neural network
    nn = NeuralNetwork(
        layers=[
            Conv(
                n_feats=12,
                filter_shape=(5, 5),
                strides=(1, 1),
                weight_scale=0.1,
                weight_decay=0.001,
            ),
            Activation('relu'),
            Pool(
                pool_shape=(2, 2),
                strides=(2, 2),
                mode='max',
            ),
            Conv(
                n_feats=16,
                filter_shape=(5, 5),
                strides=(1, 1),
                weight_scale=0.1,
                weight_decay=0.001,
            ),
            Activation('relu'),
            Flatten(),
            Linear(
                n_out=n_classes,
                weight_scale=0.1,
                weight_decay=0.02,
            ),
            LogRegression(),
        ],
    )

    # Train neural network
    t0 = time.time()
    nn.fit(X_train, y_train, learning_rate=0.05, max_iter=3, batch_size=32)
    t1 = time.time()
    print('Duration: %.1fs' % (t1-t0))
