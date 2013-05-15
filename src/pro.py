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
import multiprocessing as mp

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from operator import itemgetter

def read_data(args):
    
    srcSents = {}
    tgtSents = {}
    for sentNum, line in enumerate(open(args.srcfile, 'r')):
        line = unicode(line, 'utf-8')
        #sentId, text = line.strip().split(' ||| ')
        srcSent, tgtSent = line.strip().split(' ||| ')
        srcSents[sentNum] = srcSent
        tgtSents[sentNum] = tgtSent
        
    #for sentNum, line in enumerate(open(args.tgtfile, 'r')):
    #    tgtSents[sentNum] = line.strip()
        
    kBestSents = {}
    oneBestSents = {}
    for sentNum, line in enumerate(open(args.kbestfile, 'r')):
        line = unicode(line, 'utf-8')
        sentId, text, probs, bleuScore = line.split(' ||| ')
        #probs = [float(prob) for prob in probs.replace('p(e|f)=','').replace('p(e)=','').replace('p_lex(f|e)=','').strip().split()]
        probs = [float(prob) for prob in probs.strip().split()]
        bleuScore = float(bleuScore)
        
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
            
        oneBestSents[sentCount] = (text, probs, bleuScore)
        
    kBestSents[n] = oneBestSents
            
    assert len(kBestSents) == len(srcSents)
    assert len(kBestSents) == len(tgtSents)
    
    return srcSents, tgtSents, kBestSents
        
def get_training_set(kBestSents, tgtSents, srcSents, T, C, numProc):
    
    #get the difference in bleu values for pairs of hypothesis
    X = []
    Y = []
    for refNum, ref in tgtSents.iteritems():
        mnPairDict = {}
        ref = tgtSents[refNum]
        selected = 0
        tries = 0
        while selected != T:
            tries += 1
            if tries > 5*T: 
                #sys.stderr.write("Skipped ")
                break
            
            i = random.randint(0, len(kBestSents[refNum])-1)
            j = random.randint(0, len(kBestSents[refNum])-1)
            if i == j: continue
        
            hyp1, feat1, bleuScore1 = kBestSents[refNum][i]
            hyp2, feat2, bleuScore2 = kBestSents[refNum][j]
            delBleu = bleuScore1 - bleuScore2
        
            if abs(delBleu) < 0.05: continue
        
            mnPairDict[(refNum, i, j)] = delBleu
            selected += 1
         
        if refNum %100 == 0: sys.stderr.write(str(refNum)+' ')
    
        selectedPairs = {key: abs(delBleu) for key, delBleu in mnPairDict.iteritems()}
        for pair, delBleu in sorted(selectedPairs.items(), key=itemgetter(1), reverse=True)[:C]:
            #add new features
            (refNum, i, j) = pair
            hyp1, feat1, bleuScore1 = kBestSents[refNum][i]
            hyp2, feat2, bleuScore2 = kBestSents[refNum][j]
            
            newFeat1 = features.add_features(srcSents[refNum], hyp1) + feat1
            newFeat2 = features.add_features(srcSents[refNum], hyp2) + feat2
            featDiff = numpy.array(newFeat1)-numpy.array(newFeat2)
            signDelBleu = abs(bleuScore1-bleuScore2)/(bleuScore1-bleuScore2)
            
            X += [ featDiff , -1.*featDiff ]
            Y += [ signDelBleu, -1.*signDelBleu ]
            
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
    parser.add_argument("-c", "--ncpu", type=int, default=10, help="no. of cpus")
    
    args = parser.parse_args()
    
    sys.stderr.write("Reading data... ")
    srcSents, tgtSents, kBestSents = read_data(args)
    sys.stderr.write("read\nProducing training data... ")
    X, Y = get_training_set(kBestSents, tgtSents, srcSents, args.sample, args.topn, args.ncpu)
    
    #train logistic classifier
    sys.stderr.write("Training classifier... ")
    model = LogisticRegression()
    #model = Ridge(fit_intercept=False, alpha=1.)
    model.fit(X, Y)
    sys.stderr.write(" trained\n")
    
    weights = model.coef_[0]
    weights /= math.sqrt((weights**2).sum() + 1e-6) # normalize weight vector
    sys.stderr.write('\n'+' '.join([str(round(w,4)) for w in weights])+'\n')
    sys.stderr.write("Now reranking... ")
    rerank.rerank(srcSents, tgtSents, kBestSents, weights, args.ncpu)
    sys.stderr.write(" Done\n")
