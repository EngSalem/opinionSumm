import argparse
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser()
parser.add_argument('--keys', action='store', dest='keys',
                    help='keys in terms of text each line is a set of reviews')
parser.add_argument('--bank', action='store', dest='bank',
                    help='bank in terms of text each line is unannotated reviews')
parser.add_argument('--k', action='store', dest='k',
                    help='K nearest neighbors')
parser.add_argument('--output', action='store', dest='output', help='output reviews')
args = parser.parse_args()


def retrieve_neighbors(index, queries,bank, k=4):
    '''
    k: number of nearest neighbors , default value =4
    queries: vectors of query sentences
    index: the index of the bank
    '''
    _, I = index.search(queries, k)  # I contains the nearest neighbors' indeces
    ## get final set of sentences' indeces
    print(len(np.array(I).flatten()))
    _ixs = np.unique(np.array(I).flatten()).tolist()
    return [bank[ix] for ix in _ixs]


## define sentence encoder
sentEncoder = SentenceTransformer('distilbert-base-nli-mean-tokens')  ## encoder to encode reviews

keys = [l.strip() for l in open(args.keys,'r').readlines()]
bank = [l.strip() for l in open(args.bank,'r').readlines()]

## encode keys
encodedKeys = sentEncoder.encode(keys)  ## result is a matrix of vectors K*V where K is number of keys and V is vector
## encode banks
encodedBank = sentEncoder.encode(bank)  ## result is a matrix of vectors B*V where B is bank sentences

## Index using faiss
d=768 ## distill bert dimension
index = faiss.IndexIDMap(faiss.IndexFlatIP(d)) ## create the bank index
index.add_with_ids(encodedBank, np.array(range(0,len(bank))))
## retrieve the nearest neighbors
reviews = retrieve_neighbors(index, encodedKeys,bank, k=int(args.k))

pd.DataFrame(data={'reviews': reviews, 'summary': [''] * len(reviews)}).to_csv(args.output)
