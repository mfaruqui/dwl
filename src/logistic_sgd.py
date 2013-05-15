import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T

class LogisticRegression(object):
    
    def __init__(self, input, n_in, n_out, W=None, b=None):
        
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX), name='W', borrow=True)
        
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=numpy.zeros((n_out,), dtype=theano.config.floatX), name='b', borrow=True)
        
        #self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.p_y_given_x = T.nnet.sigmoid(T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in symbolic form
        self.y_pred = self.p_y_given_x

        # parameters of the model
        self.params = [self.W, self.b]
        
    def reset_params(self, W, b):
        
        W = theano.shared(value=W, name='W', borrow=True)
        b = theano.shared(value=b, name='b', borrow=True)
        
        self.W = W
        self.b = b
        
        self.params = [self.W, self.b]
        
    def throw_output(self, inputVector):
        
        return T.nnet.sigmoid(T.dot(inputVector, self.W) + self.b)
        
    def return_numpy_array(self, a):
        
        return_func = theano.function(inputs=[], outputs=[a])
        array = return_func()
        return array

    def errors(self, y):
        
        #Cross entropy
        L = -T.sum(y * T.log(self.y_pred) + (1 - y) * T.log(1 - self.y_pred), axis=1)
        return T.mean(L)
        
        #Euclidean Distance
        #return T.mean(T.sqrt(T.sum(T.sqr(y - self.y_pred),1)))
        
        #Manhattan Distance
        #return T.mean(T.sum(T.abs_(y - self.y_pred),1))
        
        #1 - Cosine similarity
        #mod_y = T.sqrt(T.sum(T.sqr(y), 1))
        #mod_y_pred = T.sqrt(T.sum(T.sqr(self.y_pred), 1))
        #return 1 - T.sum(T.dot(y.flatten(), self.y_pred.flatten()))/(mod_y*mod_y_pred)
        
        #1 - Cosine similarity in numpy
        