__author__ = 'diego'

"""
layer that combines the various activation functions
"""

__docformat__ = 'restructedtext en'

import numpy
import theano
import theano.tensor as T

class HiddenLayer(object):
    def __init__(self, input, n_out, n_activations, W=None, b=None):
        """
        Weight matrix W is of shape (n_in,n_out) - however many entries are zero
        and the bias vector b is of shape (n_out,).

        n_in must be: n_out * n_activations

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_out: int
        :param n_out: number of hidden units

        :type n_activations: number of activiation functions
        :param n_activations: n_in is really n_activations * true size of input
        """

        n_in = n_out*n_activations

        self.input = input

        # `W` is initialized with `W_values` that put equal weight to each activation function
        if W is None:
            v = 1.0/n_activations
            W_values = numpy.ones(shape=(n_in,n_out),dtype=theano.config.floatX)*v
            W = theano.shared(value=W_values, name='W', borrow=True)

        self.W = W

        # mask because only some connections exist
        blocks = []
        for i in range(n_activations):
            mask_values = numpy.identity(n_activations,dtype=theano.config.floatX)
            m = theano.shared(value=mask_values, name='m', borrow=True)
            blocks.append(m)
        mask = T.concatenate(blocks)

        # linear multiplication and addition
        lin_output = T.dot(input, self.W*mask)

        self.output = (lin_output)

        # parameters of the model
        self.params = [self.W]
