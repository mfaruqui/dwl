import sys
from scipy.sparse import csr_matrix
import numpy
import re
from collections import Counter

from naive_dwl import get_dicts
from naive_dwl import get_kbest_dict
from naive_dwl import normalize_word

def convert_dict_to_csr_matrix(matrixDict, sizeData, langVocab):
    
    row = numpy.zeros(len(matrixDict), dtype=int)
    col = numpy.zeros(len(matrixDict), dtype=int)
    values = numpy.zeros(len(matrixDict), dtype=int)
    
    index = 0
    for (r, c), val in matrixDict.iteritems():
        row[index] = r
        col[index] = c
        values[index] = val
        index += 1
    
    matrixLang = csr_matrix((values,(row,col)), shape=(sizeData,len(langVocab)))
    return matrixLang

def get_parallel_cooccurence_arrays(fileName, lang1Vocab, lang2Vocab):
    
    matrixDict1 = Counter()
    numLine = 0
    for line in open(fileName, 'r'):
        lang1, lang2 = line.split('|||')
        lang1 = unicode(lang1.strip().lower(), 'utf-8')
        lang2 = unicode(lang2.strip().lower(), 'utf-8')
        
        for word in lang1.split():
            word = normalize_word(word)
            if word in lang1Vocab:
                # we want count of the words on the input
                matrixDict1[(numLine,lang1Vocab[word])] += 1
                
        numLine += 1
    
    matrixLang1 = convert_dict_to_csr_matrix(matrixDict1, numLine, lang1Vocab)  
    del matrixDict1
    
    matrixDict2 = Counter()
    numLine = 0
    for line in open(fileName, 'r'):
        lang1, lang2 = line.split('|||')
        lang1 = unicode(lang1.strip().lower(), 'utf-8')
        lang2 = unicode(lang2.strip().lower(), 'utf-8')
                
        for word in lang2.split():
            word = normalize_word(word)
            if word in lang2Vocab:
                # we want probability of occurrence on the output
                matrixDict2[(numLine,lang2Vocab[word])] = 1
            
        numLine += 1
    
    matrixLang2 = convert_dict_to_csr_matrix(matrixDict2, numLine, lang2Vocab)  
    del matrixDict2
    
    return (matrixLang1, matrixLang2)
    
def get_datasets(trFile, valFile, testFile, kbestFile, freqCutoff):
    
    trainSrcDict, trainTgtDict = get_dicts(trFile)
    devSrcDict, devTgtDict = get_dicts(valFile)
    testSrcDict, testTgtDict = get_dicts(testFile)
    kbestTgtDict = get_kbest_dict(kbestFile)

    devAndTestSrcDict = {}
    for word in set(devSrcDict.keys()) | set(testSrcDict.keys()):
        if trainSrcDict[word]+devSrcDict[word]+testSrcDict[word] > freqCutoff:
            devAndTestSrcDict[word] = len(devAndTestSrcDict)

    trainTargetWords = [key for key in kbestTgtDict.iterkeys() if trainTgtDict[key]+devTgtDict[key]+testTgtDict[key] > freqCutoff]
    sys.stderr.write("Feature size: "+str(len(devAndTestSrcDict))+"\n")
    sys.stderr.write("Num target words: "+str(len(trainTargetWords))+"\n")

    trainTargetWordDict = {word:index for index, word in enumerate(trainTargetWords)}

    sys.stderr.write("\nFiles read...\n")
    sys.stderr.write("Total vocab sizes: lang1 = {0}, lang2 = {1}\n".format(len(devAndTestSrcDict), len(trainTargetWordDict)))
    
    datasets = []
    datasets.append(get_parallel_cooccurence_arrays(trFile, devAndTestSrcDict, trainTargetWordDict))
    datasets.append(get_parallel_cooccurence_arrays(valFile, devAndTestSrcDict, trainTargetWordDict))
    
    return datasets, devAndTestSrcDict, trainTargetWordDict
