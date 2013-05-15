import argparse
import numpy
import bleu
import pickle
import numpy
import sys
import random
import math
import features
import rerank

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from operator import itemgetter

def read_data(args):
    
    srcSents = {}
    tgtSents = {}
    for sentNum, line in enumerate(open(args.srcfile, 'r')):
        #sentId, text = line.strip().split(' ||| ')
        srcSent, tgtSent = line.strip().split(' ||| ')
        srcSents[sentNum] = srcSent
        tgtSents[sentNum] = tgtSent
        
    #for sentNum, line in enumerate(open(args.tgtfile, 'r')):
    #    tgtSents[sentNum] = line.strip()
        
    kBestSents = {}
    oneBestSents = {}
    for sentNum, line in enumerate(open(args.kbestfile, 'r')):
        sentId, text, probs = line.split(' ||| ')
        #probs = [float(prob) for prob in probs.replace('p(e|f)=','').replace('p(e)=','').replace('p_lex(f|e)=','').strip().split()]
        probs = [float(prob) for prob in probs.strip().split()]
        
        if sentNum == 0:
            n = 0
            prevSentId = sentId
            sentCount = 0
        elif sentId != prevSentId:
            prevSentId = sentId
            sentCount = 0
            kBestSents[n] = oneBestSents
            oneBestSents = {}
            n += 1
        else:
            sentCount += 1
            
        oneBestSents[sentCount] = (text, probs)
        
    kBestSents[n] = oneBestSents
            
    assert len(kBestSents) == len(srcSents)
    assert len(kBestSents) == len(tgtSents)
    
    return srcSents, tgtSents, kBestSents
    
def get_training_set(kBestSents, tgtSents, srcSents, T, C):
    
    #get the difference in bleu values for pairs of hypothesis
    mnPairDict = {}
    selected = 0
    while selected != T:
        refNum = random.randint(0, len(tgtSents)-1)
        ref = tgtSents[refNum]
        i = random.randint(0, len(kBestSents[refNum])-1)
        j = random.randint(0, len(kBestSents[refNum])-1)
        if i == j: continue
        
        hyp1, feat1 = kBestSents[refNum][i]
        hyp2, feat2 = kBestSents[refNum][j]
        delBleu = bleu.bleu_pair(hyp1, ref) - bleu.bleu_pair(hyp2, ref)
        if abs(delBleu) < 0.05: continue
        
        mnPairDict[(refNum, i, j)] = delBleu
        selected += 1
        if selected %1000 == 0: sys.stderr.write(str(selected)+' ')
    
    X = []
    Y = []            
    selectedPairs = {key: abs(delBleu) for key, delBleu in mnPairDict.iteritems()}
    selected = 0
    for pair, delBleu in sorted(selectedPairs.items(), key=itemgetter(1), reverse=True):
        
        #add new features
        (refNum, i, j) = pair
        hyp1, feat1 = kBestSents[refNum][i]
        hyp2, feat2 = kBestSents[refNum][j]
        
        newFeat1 = features.add_features(srcSents[refNum], hyp1) + feat1
        newFeat2 = features.add_features(srcSents[refNum], hyp2) + feat2
        featDiff = numpy.array(newFeat1)-numpy.array(newFeat2)
        
        X += [ featDiff , -1.*featDiff ]
        Y += [ abs(mnPairDict[pair])/mnPairDict[pair], -1.*abs(mnPairDict[pair])/mnPairDict[pair] ]
        
        selected += 1
        if selected == C:
            break
            
    del selectedPairs, mnPairDict
    return (X, Y)
            
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-src", "--srcfile", type=str, help="Source file")
    parser.add_argument("-tgt", "--tgtfile", type=str, help="Target file")
    parser.add_argument("-kbest", "--kbestfile", type=str, default=None, help="k-best translations of each source sentence")
    parser.add_argument("-t", "--sample", type=int, default=5000, help="sample t best pairs according to abs(delBleu)")
    parser.add_argument("-n", "--topn", type=int, default=50, help="choose top n pairs from the sampled ones")
    parser.add_argument("-o", "--clasname", type=str, default="clas.pickle", help="classifier file name")
    
    args = parser.parse_args()
    
    sys.stderr.write("Reading data... ")
    srcSents, tgtSents, kBestSents = read_data(args)
   
    for refNum, ref in sorted(tgtSents.items(), key=itemgetter(0)):
        for transNum, trans in sorted(kBestSents[refNum].items(), key=itemgetter(0)):
            hyp, feat = trans
            bleuScore = bleu.bleu_pair(hyp, ref)
            print str(refNum)+' ||| '+hyp+' ||| '+' '.join([str(f) for f in feat])+' ||| '+str(bleuScore)
