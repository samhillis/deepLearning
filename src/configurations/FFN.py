__author__ = 'diego'

"""
class for deep feed forward network
"""

import theano.tensor as T

from src.layersFFN.hidden import HiddenLayer
from src.layersFFN.softMax import SoftMax

class FFN(object):
    """
    Feedforward artificial neural network model
    that has several hidden layers and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``SoftMax``
    class).
    """

    def __init__(self, rng, input, layers_hidden, n_in, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type layers_hidden: List[int]
        :param layers_hidden: number of hidden units in each hidden layer; the number of hidden layers matches len(layers_hidden)

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie
        """

        # all hidden layers
        self.hiddenLayers = []
        previous_output = input
        n_input = n_in
        for n_hidden in layers_hidden:
            new_layer = HiddenLayer(
                rng=rng,
                input=previous_output,
                n_in= n_input,
                n_out=n_hidden,
                activation=T.tanh
            )
            n_input = n_hidden
            previous_output = new_layer.output
            self.hiddenLayers.append(new_layer)

        # The logistic regression (softmax) layer
        self.logRegressionLayer = SoftMax(
            input=self.hiddenLayers[-1].output,
            n_in=layers_hidden[-1],
            n_out=n_out
        )

        # L1 norm ; one regularization option is to enforce L1 norm to be small
        l1reg = abs(self.logRegressionLayer.W).sum()
        for i in self.hiddenLayers:
            l1reg += abs(i.W).sum()
        self.L1 = (l1reg)

        # square of L2 norm ; one regularization option is to enforce square of L2 norm to be small
        l2reg = (self.logRegressionLayer.W ** 2).sum()
        for i in self.hiddenLayers:
            l2reg += (i.W ** 2).sum()
        self.L2_sqr = (l2reg)

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the layers
        self.params = []
        for p in self.hiddenLayers:
            self.params += p.params
        self.params += self.logRegressionLayer.params

        # keep track of model input
        self.input = input
