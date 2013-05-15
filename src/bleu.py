import math
from collections import Counter

# written by Adam Lopez

# Collect BLEU-relevant statistics for a single hypothesis/reference pair.
# Return value is a generator yielding:
# (c, r, numerator1, denominator1, ... numerator4, denominator4)
# Summing the columns across calls to this function on an entire corpus will
# produce a vector of statistics that can be used to compute BLEU (below)
def bleu_stats(hypothesis, reference):
  yield len(hypothesis)
  yield len(reference)
  for n in xrange(1,5):
    s_ngrams = Counter([tuple(hypothesis[i:i+n]) for i in xrange(len(hypothesis)+1-n)])
    r_ngrams = Counter([tuple(reference[i:i+n]) for i in xrange(len(reference)+1-n)])
    yield max([sum((s_ngrams & r_ngrams).values()), 0])
    yield max([len(hypothesis)+1-n, 0])

# Compute BLEU from collected statistics obtained by call(s) to bleu_stats
def bleu(stats):
  if len(filter(lambda x: x==0, stats)) > 0:
    return 0
  (c, r) = stats[:2]
  log_bleu_prec = sum([math.log(float(x)/y) for x,y in zip(stats[2::2],stats[3::2])]) / 4.
  return math.exp(min([0, 1-float(r)/c]) + log_bleu_prec)
  
#written by Manaal 
#computes BLEU for a given reference and hypothesis
def bleu_pair(hyp, ref):
    stats = [0 for i in xrange(10)]
    #stats = [0,0,1,1,1,1,1,1,1,1]
    stats = [sum(scores) for scores in zip(stats, bleu_stats(hyp,ref))]
    
    #return smoothed bleu
    return bleu(stats)
    
if __name__=='__main__':
    print bleu_pair("I love you", "I love you")
