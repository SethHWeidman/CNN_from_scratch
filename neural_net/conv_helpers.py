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


class Flatten(Layer):
    def fprop(self, input):
        self.last_input_shape = input.shape
        return np.reshape(input, (input.shape[0], -1))

    def bprop(self, output_grad):
        return np.reshape(output_grad, self.last_input_shape)

    def output_shape(self, input_shape):
        return (input_shape[0], np.prod(input_shape[1:]))