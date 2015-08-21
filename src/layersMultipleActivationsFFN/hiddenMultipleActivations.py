__author__ = 'diego'

"""
hidden full connected layer with multiple activations; for each activation function there is a copy of the output nodes
all copies share the same bias
"""

__docformat__ = 'restructedtext en'

import numpy
import theano
import theano.tensor as T

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activations=[T.tanh]):
        """
        Weight matrix W is of shape (n_in,n_out*len(activations))
        and the bias vector b is of shape (n_out,).

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activations: list of theano.Op's or functions
        :param activation: Non linearity to be applied in the hidden layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        if W is None:
            subArrays = []
            for actFunction in activations:
                W_values = numpy.asarray(
                    rng.uniform(
                        low=-numpy.sqrt(6. / (n_in + n_out)),
                        high=numpy.sqrt(6. / (n_in + n_out)),
                        size=(n_in, n_out)
                    ),
                    dtype=theano.config.floatX
                )
                if actFunction == theano.tensor.nnet.sigmoid:
                    W_values *= 4
                subArrays.append(W_values)

            # concatenate them together
            numpy.concatenate(subArrays,axis=1)

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        # linear multiplication and addition
        lin_output = T.dot(input, self.W) + self.b

        # apply activation functions
        index = 0
        subOutput = []
        for actFunction in activations:
            subOutput.append(actFunction(lin_output[index:index+n_out]))
            index += n_out

        self.output = (T.concatenate(subOutput))

        # parameters of the model
        self.params = [self.W, self.b]
