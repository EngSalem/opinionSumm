import argparse
import pandas as pd
from itertools import permutations, combinations, combinations_with_replacement
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-infile', type=str)
parser.add_argument('-outfile', type=str)
parser.add_argument('-k', type=int)
args = parser.parse_args()


def delete_random_elems(input_list, n):
    to_delete = set(random.sample(range(len(input_list)), n))
    return [x for i, x in enumerate(input_list) if not i in to_delete]


def shuffle_and_delete(reviews, k=2):

    random.shuffle(reviews)
    return ' '.join(delete_random_elems(reviews, k))


df = pd.read_csv(args.infile, delimiter='\t')
df['grouped_reviews']= df[['rev' + str(i) for i in range(1, 9)]].values.tolist()
df['grouped_summaries']= df[['summ1','summ2','summ3']].values.tolist()


df['review'] = df.apply(lambda row: shuffle_and_delete(row['grouped_reviews'], k=args.k), axis=1)

group_ids = []
reviews = []
summaries = []

for group_id, rev, summ in zip(df.group_id.tolist(), df.review.tolist(), df.grouped_summaries.tolist()):
    summaries.extend(summ)
    reviews.extend([rev]*3)
    group_ids.extend([group_id]*3)



pd.DataFrame(data={'group_id': group_ids, 'review': reviews, 'summary':summaries}).to_csv(args.outfile, index=False)
