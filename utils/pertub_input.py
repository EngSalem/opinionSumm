import argparse
import pandas as pd
from itertools import permutations, combinations, combinations_with_replacement
import random

def delete_random_elems(input_list, n):
    to_delete = set(random.sample(range(len(input_list)), n))
    return [x for i,x in enumerate(input_list) if not i in to_delete]

def shuffle_and_delete(reviews, k=2):
    random.shuffle(reviews)
    return ' '.join(delete_random_elems(reviews,k))


def get_reviews(df):
    group_ids = []
    reviews = []
    df['grouped_reviews']= df[['rev' + str(i) for i in range(1, 9)]].values.tolist()
    df['grouped_summaries']= df[['summ1','summ2','summ3']].values.tolist()

    group_ids, summaries, reviews = [], [], []
    for grp_id, df_grp in df.groupby(by=['group_id']):
        ## add initial input
        group_ids.extend([grp_id]*3)
        summaries.extend(df_grp['grouped_summaries'].values[0])

        reviews.extend([' '.join(df_grp['grouped_reviews'].values[0])]*3)
        for k in range(1,4):
            group_ids.extend([grp_id+'_k'+str(k)]*3)
            reviews.extend([shuffle_and_delete(df_grp['grouped_reviews'].values[0], k)]*3)
            summaries.extend(df_grp['grouped_summaries'].values[0])

    return pd.DataFrame(data={'group_id': group_ids,'review': reviews, 'summary': summaries})








## test reviews
## amazon reviews
amazon_test_df = pd.read_csv('../FewSumm/artifacts/amazon/gold_summs/test.csv', delimiter='\t')
## yelp reviews
yelp_test_df = pd.read_csv('../FewSumm/artifacts/yelp/gold_summs/test.csv', delimiter='\t')

amazon_test_df = get_reviews(amazon_test_df)
yelp_test_df = get_reviews(yelp_test_df)

amazon_test_df.to_csv('../FewSumm/artifacts/amazon/gold_summs/sum_pairs/amazon_test_df_multiple_inputs.csv', index=False)
yelp_test_df.to_csv('../FewSumm/artifacts/amazon/gold_summs/sum_pairs/yelp_test_df_multiple_inputs.csv', index=False)