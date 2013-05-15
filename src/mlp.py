import theano
import theano.tensor as T
import numpy
import sys

from logistic_sgd import LogisticRegression
from hidden_layer import HiddenLayer

class MLP(object):
    
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        
        self.hiddenLayer = HiddenLayer(rng=rng, input=input,
                                       n_in=n_in, n_out=n_hidden,
                                       activation=T.tanh)

        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out)

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = abs(self.hiddenLayer.W).sum() + abs(self.logRegressionLayer.W).sum()

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (self.hiddenLayer.W ** 2).sum() + (self.logRegressionLayer.W ** 2).sum()

        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        
    def save_model_params(self, outFile):
        
        sys.stderr.write("\nSaving model params to file... ")
        return_params = theano.function(inputs=[], outputs=self.params)
        model_params = return_params()
        numpy.savez(outFile, model_params)
        sys.stderr.write("saved\n")
        
    def load_model_params(self, paramList):
        
        W1, b1, W2, b2 = paramList
        self.hiddenLayer.reset_params(W1, b1)
        self.logRegressionLayer.reset_params(W2, b2)
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        
    def throw_output(self, inputVector):
        
        hiddenOut = self.hiddenLayer.throw_output(inputVector)
        logitOut = self.logRegressionLayer.throw_output(hiddenOut)
        return_params = theano.function(inputs=[], outputs=logitOut)
        return return_params()
        