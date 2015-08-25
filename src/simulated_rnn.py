__author__ = 'diego'

import numpy
import theano
import theano.tensor as T

class HiddenLayer(object):
    def __init__(self, rng, input, n_inh, n_ini, n_out, Wh=None, Wi=None, b=None,
                 activation=T.tanh):
        """
        :param rng: random number stream
        :param input: tuple of tensors; first element corresponds to the hidden layer, the second element to the input layer
        :param n_inh: input of hidden layer
        :param n_ini: input from input layer
        :param n_out: number of nodes in output layer
        :param Wh: parameters for hidden layer
        :param Wi: parameters for input layer
        :param b: bias
        :param activation: the activation function
        """

        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        def init_weights(W,n_in,name):
            if W is None:
                W_values = numpy.asarray(
                    rng.uniform(
                        low=-numpy.sqrt(6. / (n_in + n_out)),
                        high=numpy.sqrt(6. / (n_in + n_out)),
                        size=(n_in, n_out)
                    ),
                    dtype=theano.config.floatX
                )
                if activation == T.nnet.sigmoid:
                    W_values *= 4

                W = theano.shared(value=W_values, name=name, borrow=True)

            return W

        Wh = init_weights(Wh,n_inh,"Wh")
        Wi = init_weights(Wi,n_ini,"Wi")

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        
        
        try:
            lin_output = T.dot(input[0], Wh) + T.dot(input[1], Wi) + b
        except:
            import pdb
            pdb.set_trace()
        
        #lin_output = T.dot(input[0], Wi) + b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

        # parameters of the model
        self.params = [Wh, Wi, b]
        #self.params = [Wi, b]

class SoftMax(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so thatcost
              the learning rate is less dependent on the batch size
        """

        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class Model(object):
    def __init__(self,rng,input,labels,n_in,n_out,n_hidden,n_records):
        """
        :param rng: random number stream
        :param input: list of input tensors; each tensor corresponds to one time unit: dim = num records x num features per time unit
        :param labels: for each time unit the tensor of labels
        :param n_in: features per time period
        :param n_out: number of output nodes per time period
        :param n_hidden: number of features in the hidden layer
        :param n_records: number of records
        """

        self.hiddenLayers = []
        self.outputLayers = []

        # the first fake hidden layer: everything is set to 0
        z1 = theano.shared(
            value=numpy.zeros(
                (n_hidden,n_hidden),
                dtype=theano.config.floatX
            ),
            name='initHH',
            borrow=True
        )
        z2 = theano.shared(
            value=numpy.zeros(
                (n_records,n_in),
                dtype=theano.config.floatX
            ),
            name='initHI',
            borrow=True
        )
        hl = HiddenLayer(rng,(z1,z2),n_hidden,n_in,n_hidden)
        self.hiddenLayers.append(hl)
        
        # form the network
        for tp in range(1,len(input)):
            previous_hidden = self.hiddenLayers[-1].output
            hl = HiddenLayer(rng,(previous_hidden,input['x'+str(tp)]),n_hidden,n_in,n_hidden)
            out = SoftMax(hl.output,n_hidden,n_out)
            self.hiddenLayers.append(hl)
            self.outputLayers.append(out)
        
        # set all parameters
        self.params = []
        for l in self.hiddenLayers:
            for w in l.params:
                self.params.append(w)
        #for l in self.outputLayers:
        #    for w in l.params:
        #        self.params.append(w)
        
    
    def negative_log_likelihood(self, labels):
        log_lkh = 0
        for i,l in enumerate(self.outputLayers):
            log_lkh = l.negative_log_likelihood(labels['y'+str(i)])
        return log_lkh/len(labels.keys())
#         log_lkh= [l.negative_log_likelihood(labels['y'+str(i)]) for i,l in enumerate(self.outputLayers)]
#         return T.sum(log_lkh)/len(labels.keys())
    
    def errors(self, labels):
        errs = []
        for i,l in enumerate(self.outputLayers):
            errs.append(l.errors(labels['y'+str(i)]))
        return errs
    
    def preds(self):
        vals = [ol.y_pred for ol in self.outputLayers]
        return vals
