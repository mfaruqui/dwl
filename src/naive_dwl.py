import sys
import numpy
import pickle
import multiprocessing as mp
import math
import argparse
import os
import json
import re

from operator import itemgetter
from scipy.sparse import coo_matrix, vstack
from collections import Counter
from sklearn.linear_model import LogisticRegression

global X, devAndTestSrcDict, sizeTrData, OUTPUT_DIR

isPunc = re.compile(r'^[:;!\?\%\$#\*\"\(\)\[\]\/,\.]$')
isNumber = re.compile(r'^\d+\.?\d*$')

def normalize_word(word):
    
    if isNumber.search(word):
        return '---NUM---'
    elif isPunc.search(word):
        return '---PUNC---'
    else:
        return word

def get_kbest_dict(fileName):

    tgtDict = Counter()
    errorLines = 0

    for line in open(fileName, 'r'):
        line = unicode(line, 'utf-8')
        try:
            lineNum, tgt, rest = line.strip().split('|||')
        except:
            errorLines += 1
            continue

        for word in tgt.strip().split():
            tgtDict[normalize_word(word)] += 1

    sys.stderr.write("File "+fileName+" processed with "+str(errorLines)+" erroneous lines\n")
    return tgtDict

def get_dicts(fileName):
    
    srcDict = Counter()
    tgtDict = Counter()
    errorLines = 0
    
    for line in open(fileName, 'r'):
        line = unicode(line, 'utf-8') 
        try:
            src, tgt = line.strip().split('|||')
        except:
            errorLines += 1
            continue
        
        for word in src.strip().split():
            srcDict[normalize_word(word)] += 1
            
        for word in tgt.strip().split():
            tgtDict[normalize_word(word)] += 1
        
    sys.stderr.write("File "+fileName+" processed with "+str(errorLines)+" erroneous lines\n")
    return srcDict, tgtDict

def get_numlines(fileName):
    
    numLines = 0
    for line in open(fileName, 'r'):
        numLines += 1
        
    return 1.*numLines

def set_src_feat_vector():
    
    global X, sizeTrData, devAndTestSrcDict, TRAIN_FILE
    
    sizeTrData = get_numlines(TRAIN_FILE)
    X1 = numpy.zeros((math.ceil(sizeTrData/2), len(devAndTestSrcDict)), dtype=int)
    
    for numLine, line in enumerate(open(TRAIN_FILE, 'r')):
        line = unicode(line, 'utf-8')
        if numLine == math.ceil(sizeTrData/2):
            break
        
        src, tgt = line.strip().split('|||')
        for word in src.split():
            word = normalize_word(word)
            if word in devAndTestSrcDict:
                X1[numLine][devAndTestSrcDict[word]] += 1
                
    X1 = coo_matrix(X1)
    X2 = numpy.zeros((sizeTrData-math.ceil(sizeTrData/2), len(devAndTestSrcDict)), dtype=int)
    
    for numLine, line in enumerate(open(TRAIN_FILE, 'r')):
        line = unicode(line, 'utf-8')
        if numLine >= math.ceil(sizeTrData/2):
            src, tgt = line.strip().split('|||')
            for word in src.split():
                word = normalize_word(word)
                if word in devAndTestSrcDict:
                    X2[numLine-math.ceil(sizeTrData/2)][devAndTestSrcDict[word]] += 1
                
    X2 = coo_matrix(X2)
    X = vstack([X1, X2])

def train_dwl(targetWord):
    
    global X, TRAIN_FILE, sizeTrData, OUTPUT_DIR
    
    Y = numpy.zeros((sizeTrData), dtype=int)
    shouldTrain = False
    
    for lineNum, line in enumerate(open(TRAIN_FILE, 'r')):
        line = unicode(line, 'utf-8')
        src, tgt = line.strip().split('|||')
        tgtWords = [normalize_word(word) for word in tgt.strip().split()]
        if targetWord in tgtWords:
            shouldTrain = True
            Y[ lineNum ] = 1
            
    if not shouldTrain:
        return

    try:
        outModelFile = open(OUTPUT_DIR+'word_'+targetWord+'.pickle', 'w')
        clf = LogisticRegression()
        clf.fit(X,Y)
        pickle.dump(clf, outModelFile)
    except:
        pass

if __name__=='__main__':

    global OUTPUT_DIR, devAndTestSrcDict
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--trainfile", type=str, help="Joint parallel file of two languages; sentences separated by |||")
    parser.add_argument("-val", "--valfile", type=str, help="Validation corpus in the same format as training file")
    parser.add_argument("-test", "--testfile", type=str, default=None, help="Test corpus in the same format as training file")
    parser.add_argument("-kbest", "--kbestfile", type=str, default=None, help="k-best corpus with text as the second field")
    parser.add_argument("-n", "--numcores", type=int, default=5, help="No. of cores to be used")
    parser.add_argument("-f", "--freq", type=int, default=10, help="Frequency cutoff")
    parser.add_argument("-odir", "--outputdir", type=str, default='data/models', help="Output Dir Name")

    args = parser.parse_args()
    
    TRAIN_FILE = args.trainfile
    DEV_FILE = args.valfile
    TEST_FILE = args.testfile
    K_BEST_FILE = args.kbestfile
    NUM_PROC = args.numcores
    FREQ_CUTOFF = args.freq
    OUTPUT_DIR = args.outputdir

    if not OUTPUT_DIR.endswith('/'):
        OUTPUT_DIR += '/'

    trainSrcDict, trainTgtDict = get_dicts(TRAIN_FILE)
    devSrcDict, devTgtDict = get_dicts(DEV_FILE)
    testSrcDict, testTgtDict = get_dicts(TEST_FILE)
    kbestTgtDict = get_kbest_dict(K_BEST_FILE)
    
    devAndTestSrcDict = {}
    for word in set(devSrcDict.keys()) | set(testSrcDict.keys()):
        if trainSrcDict[word]+devSrcDict[word]+testSrcDict[word] > FREQ_CUTOFF:
            devAndTestSrcDict[word] = len(devAndTestSrcDict)

    trainTargetWords = [key for key in kbestTgtDict.iterkeys() if trainTgtDict[key]+devTgtDict[key]+testTgtDict[key] > FREQ_CUTOFF]
    sys.stderr.write("Feature size: "+str(len(devAndTestSrcDict))+"\n")
    sys.stderr.write("Num target words: "+str(len(trainTargetWords))+"\n")
    
    trainTargetWordDict = {word:index for index, word in enumerate(trainTargetWords)}

    with open(OUTPUT_DIR+'vocab', 'w') as vocabFile:
        vocabFile.write(json.dumps(devAndTestSrcDict).encode('utf-8')+'\n')
        vocabFile.write(json.dumps(trainTargetWordDict).encode('utf-8'))
        
    del trainTargetWordDict

    pool = mp.Pool(NUM_PROC, set_src_feat_vector)
    pool.map(train_dwl, trainTargetWords)
