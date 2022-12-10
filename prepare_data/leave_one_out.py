import pandas as pd
import argparse
from random import sample
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
MAX_SAMPLE_ITERS = 10
N_reviews = 8
#########################
parser = argparse.ArgumentParser()
parser.add_argument('--review_chunk', action='store', dest='review_chunk',
                    help='')
parser.add_argument('--out_review_summ', action='store', dest='out_reviews_summs',
                    help='')

args = parser.parse_args()
##########################

reviews = pd.read_csv(args.review_chunk)[['business_id', 'text']].rename(columns={'text': 'review'})
reviews_counts = reviews.groupby(['business_id'])['business_id'].count().sort_values(ascending=False)
## get reviews with sufficient number of reviews
rev_count_df = pd.DataFrame.from_dict({'business_id': reviews_counts.index.tolist(), 'count': reviews_counts.tolist()})
rev_count_df = rev_count_df[rev_count_df['count'] >= 9]

rev_count_df['business_id'] = [str(b_id) for b_id in rev_count_df['business_id'].tolist()]
reviews['business_id'] = [str(b_id) for b_id in reviews['business_id'].tolist()]

###
bus_ids, revs, summs = [], [], []
for b_id, rev_df in reviews.merge(rev_count_df, on='business_id', how='inner').groupby(by=['business_id']):
    bus_ids.append(b_id)

    curr_reviews = rev_df['review'].tolist()  ## full review list
    # sample one out
    summ_rev = sample(rev_df['review'].tolist(), 1)[0]

    curr_reviews.remove(summ_rev)
    ## select reviews such that they maximize the rouge-1 score between selected review and candidat references
    # revs.append('\n'.join(sample(rev_df['review'].tolist(),8)))
    ## iterating and sampling to avoid brute force search for summaries (for faster computation)
    max_rouge = 0.
    selected_revs = None
    for _ in range(MAX_SAMPLE_ITERS):
        candidate_rev = '\n'.join(sample(curr_reviews, N_reviews))
        ## compute rouge
        if scorer.score(candidate_rev,summ_rev)['rouge1'].fmeasure > max_rouge:
           selected_revs  = candidate_rev
           max_rouge = scorer.score(candidate_rev,summ_rev)['rouge1'].fmeasure

    revs.append(selected_revs)
    summs.append(summ_rev)

pd.DataFrame.from_dict({'business_id':bus_ids, 'reviews': revs, 'summary': summs}).to_csv(args.out_reviews_summs)
