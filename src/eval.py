import os
import sys
import math
import numpy
import argparse

import theano
import theano.tensor as T

from mlp import MLP

from process_parallel_data import get_datasets

def load_classifier(paramList):
    
    W1, b1, W2, b2 = paramList
    
    n_input = len(W1)
    n_hidden = len(W2)
    n_out = len(W2[0])
    x = T.matrix('x')
    rng = numpy.random.RandomState(1234)
    
    classifier = MLP(rng=rng, input=x, n_in=n_input, n_hidden=n_hidden, n_out=n_out)
    classifier.load_model_params(paramList)
    
    return classifier

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--trainfile", type=str, help="Joint parallel file of two languages; sentences separated by |||")
    parser.add_argument("-val", "--valfile", type=str, help="Validation corpus in the same format as training file")
    parser.add_argument("-param", "--paramfile", type=str, help="File with parameters of the model")
    
    args = parser.parse_args()
    
    trainFileName = args.trainfile
    valFileName = args.valfile
    paramFileName = args.paramfile
    
    datasets = get_datasets(trainFileName, valFileName)
    valid_set_x, valid_set_y = datasets[1]
    del datasets
    
    paramList = numpy.load(open(paramFileName, 'r'))
    classifier = load_classifier(paramList['arr_0'])
    
    for input_x in valid_set_x:
        output = classifier.throw_output(input_x)
        print output
        break