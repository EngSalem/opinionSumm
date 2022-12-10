import pandas as pd
import argparse
from random import sample


parser = argparse.ArgumentParser()
parser.add_argument('--review_chunk', action='store', dest='review_chunk',
                    help='')
parser.add_argument('--out_reviews', action='store', dest='out_reviews',
                    help='')

args = parser.parse_args()

reviews = pd.read_csv(args.review_chunk)[['business_id','text']].rename(columns={'text':'review'})
reviews_counts = reviews.groupby(['business_id'])['business_id'].count().sort_values(ascending=False)

rev_count_df = pd.DataFrame.from_dict({'business_id': reviews_counts.index.tolist(), 'count': reviews_counts.tolist()})
rev_count_df = rev_count_df[rev_count_df['count'] >= 8]
rev_count_df['business_id'] = [str(b_id) for b_id in rev_count_df['business_id'].tolist()]
reviews['business_id'] = [str(b_id) for b_id in reviews['business_id'].tolist()]

bus_ids,revs = [],[]
for b_id, rev_df in reviews.merge(rev_count_df, on='business_id', how='inner').groupby(by = ['business_id']):
    bus_ids.append(b_id)
    revs.append('\n'.join(sample(rev_df['review'].tolist(),8)))

pd.DataFrame.from_dict({'business_id':bus_ids, 'reviews': revs}).to_csv(args.out_reviews)

