import sys
import math
import re
import json
import numpy
import theano.tensor as T
import pickle

from mlp import MLP
from process_parallel_data import normalize_word

PUNCT = re.compile(r'^[:;!\?\%\$#\*\"\(\)\[\]\/,\.]$')
NUM = re.compile(r'^\d+\.?\d*$')
STOP = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', '\'s', 'n\'t', '\'d', 'can', 'will', 'just', 'should', 'now'])
ASCII = set(''.join(chr(i) for i in range(128)))

def load_vocab_files(vocabFileName):

    with open(vocabFileName, 'r') as vocabFile:
        src_vocab = json.loads(unicode(vocabFile.readline(), 'utf-8'))
        tgt_vocab = json.loads(unicode(vocabFile.readline(), 'utf-8'))
        vocabFile.close()

    return src_vocab, tgt_vocab

def load_nn_dwl(paramFileName):

    paramList = numpy.load(open(paramFileName, 'r'))
    W1, b1, W2, b2 = paramList['arr_0']
    n_input = len(W1)
    n_hidden = len(W2)
    n_out = len(W2[0])
    x = T.matrix('x')
    rng = numpy.random.RandomState(1234)

    classifier = MLP(rng=rng, input=x, n_in=n_input, n_hidden=n_hidden, n_out=n_out)
    classifier.load_model_params(paramList['arr_0'])

    return classifier

def load_naive_dwl(naiveDwlDir, vocab):

    dwl = {}
    sys.stderr.write('Loading naive dwl... ')
    for word in vocab:
        fileName = naiveDwlDir+'word_'+word+'.pickle'
        try:
            dwl[word] = pickle.load(open(fileName, 'r'))
        except:
            sys.stderr.write('.')

    sys.stderr.write(' done\n')
    return dwl

OOV_LOG_PROB = -10.
#NAIVE_DWL_DIR = 'data/naive_dwl/zh-en/'
#NAIVE_DWL_SRC_VOCAB, NAIVE_DWL_TGT_VOCAB = load_vocab_files(NAIVE_DWL_DIR+'vocab')
#NAIVE_DWL = load_naive_dwl(NAIVE_DWL_DIR, NAIVE_DWL_TGT_VOCAB)

nn_dwl_vocab = 'data/nn_dwl/zh-en/f10_b50_n10_h100_r0.05_vocab'
nn_dwl_param = 'data/nn_dwl/zh-en/f10_b50_n10_h100_r0.05.npz'
NN_DWL = load_nn_dwl(nn_dwl_param)
NN_DWL_SRC_VOCAB, NN_DWL_TGT_VOCAB = load_vocab_files(nn_dwl_vocab)

def add_features(s, t):
    
    #return []
    #return [tgtsrc_len_ratio(s,t), len_target(s,t)]
    #return [nn_dwl(s,t)]#, tgtsrc_len_ratio(s,t), len_target(s,t)]
    #return [nn_dwl(s,t)]
    phraseScore, oovScore = nn_dwl(s,t)
    return [phraseScore, oovScore, len_target(s,t)]

def log_tgtsrc_len_ratio(source, target):
    return math.log(1.*len(target.split())/len(source.split()))

def tgtsrc_len_ratio(source, target):
    return 1.*len(target.split())/len(source.split())

def len_target(source, target):
    return len(target.split())
    
def tgt_content(source, target):
    return sum(w not in STOP for w in target)
    
def tgt_punc(source, target):
    return sum(bool(PUNCT.match(w)) for w in target)
    
def tgt_numbers(source, target):
    return sum(bool(NUM.match(w)) for w in target)
    
def tgt_nonascii(source, target):
    return sum(bool(any(c not in ASCII for c in tgtWord)) for tgtWord in target.split())
    
def naive_dwl(source, target):
    
    X = numpy.zeros((len(NAIVE_DWL_SRC_VOCAB)), dtype=int)
    for srcWord in source.split():
         srcWord = normalize_word(srcWord)
         if srcWord in NAIVE_DWL_SRC_VOCAB:
             X[ NAIVE_DWL_SRC_VOCAB[srcWord] ] += 1
            
    phraseScore = 0.
    numOOV = 0
    for word in target.split():
        word = normalize_word(word)
        if word in NAIVE_DWL:
            probs = NAIVE_DWL[word].predict_proba(X)
            if probs[0][1] == 0:
               phraseScore += 0.
            elif probs[0][0] == 0:
               phraseScore += 0.
            else:
               phraseScore += math.log(probs[0][1]) - math.log(probs[0][0])
        else:
            numOOV += 1
              
    return phraseScore/(len(target.split())-numOOV), numOOV*OOV_LOG_PROB/len(target.split())
    
def nn_dwl(source, target):

    X = numpy.zeros((len(NN_DWL_SRC_VOCAB)), dtype=int)
    for srcWord in source.split():
         srcWord = normalize_word(srcWord)
         if srcWord in NN_DWL_SRC_VOCAB:
             X[ NN_DWL_SRC_VOCAB[srcWord] ] += 1
    
    y_pred = NN_DWL.throw_output(X)
    phraseScore = 0.
    numOOV = 0
    for word in target.split():
        word = normalize_word(word)
        if word in NN_DWL_TGT_VOCAB:
            phraseScore += math.log(y_pred[ NN_DWL_TGT_VOCAB[word] ]/(1 - y_pred[ NN_DWL_TGT_VOCAB[word] ]))
        else:
            numOOV += 1
    
    return phraseScore/(len(target.split())-numOOV), numOOV*OOV_LOG_PROB/len(target.split())
    #return sent_log_prob
