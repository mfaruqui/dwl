import numpy
import theano
import theano.tensor as T

class HiddenLayer(object):
    
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        
        self.input = input

        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            
        W = theano.shared(value=W_values, name='W', borrow=True)
        b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        
        self.params = [self.W, self.b]
        
    def reset_params(self, W, b):
        
        W = theano.shared(value=W, name='W', borrow=True)
        b = theano.shared(value=b, name='b', borrow=True)
        
        self.W = W
        self.b = b
        
        self.params = [self.W, self.b]
        
    def throw_output(self, inputVector, activation=T.tanh):
        
        lin_output = T.dot(inputVector, self.W) + self.b
        output = (lin_output if activation is None else activation(lin_output))
                       
        return output
