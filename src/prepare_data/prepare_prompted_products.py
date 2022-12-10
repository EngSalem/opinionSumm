import pandas as pd
import os

amazon_df = '../../../FewSumm/artifacts/amazon/gold_summs'
yelp_df = '../../../FewSumm/artifacts/yelp/gold_summs'


def get_rev_summ_pairs(df):
    rev_id, reviews, summaries =[], [],[]
    for rev_grp_id, rev_df in df.groupby(by=['group_id']):
        full_rev, full_summ = '', ''
        for rev_ix in range(1,9):
            full_rev = '\n'.join([full_rev, rev_df['rev' + str(rev_ix)].tolist()[0]])
        full_rev = ' '.join([rev_df.prompts.tolist()[0], full_rev])
        for summ_ix in range(1,4):
            full_summ =  rev_df['summ'+str(summ_ix)].tolist()[0]
            rev_id.append(rev_grp_id)
            reviews.append(full_rev)
            summaries.append(full_summ)

    return pd.DataFrame(data={'review_id': rev_id, 'review': reviews, 'summary':summaries})

def get_rev_summ_pairs_test(df):
    rev_id, reviews, summaries = [], [], []
    for rev_grp_id, rev_df in df.groupby(by=['group_id']):
        full_rev, full_summ = '', ''
        for rev_ix in range(1, 9):
            full_rev = '\n'.join([full_rev, rev_df['rev' + str(rev_ix)].tolist()[0]])

        full_rev = ' '.join([df.prompts.tolist()[0], full_rev])
        tmp_summs = []
        for summ_ix in range(1, 4):
            tmp_summs.append(rev_df['summ' + str(summ_ix)].tolist()[0])

        rev_id.append(rev_grp_id)
        reviews.append(full_rev)
        summaries.append(tmp_summs)

    return pd.DataFrame(data={'review_id':rev_id, 'review':reviews, 'summary':summaries})



## read amazon
amazon_train_df = pd.read_csv(os.path.join(amazon_df, 'train.csv'), delimiter='\t')
amazon_valid_df = pd.read_csv(os.path.join(amazon_df, 'val.csv'), delimiter='\t')
amazon_test_df = pd.read_csv(os.path.join(amazon_df, 'test.csv'), delimiter='\t')

## read yelp
yelp_train_df = pd.read_csv(os.path.join(yelp_df, 'train.csv'), delimiter='\t')
yelp_valid_df = pd.read_csv(os.path.join(yelp_df, 'val.csv'), delimiter='\t')
yelp_test_df = pd.read_csv(os.path.join(yelp_df, 'test.csv'), delimiter='\t')

## combine reviews and summaries amazon
#get_rev_summ_pairs(amazon_train_df).to_csv(amazon_df +'/sum_pairs/amazon_train_df_kw_prompted.csv', index=False)
#get_rev_summ_pairs(amazon_valid_df).to_csv(amazon_df + '/sum_pairs/amazon_valid_df_kw_prompted.csv', index=False)
#get_rev_summ_pairs(amazon_test_df).to_csv(amazon_df+ '/sum_pairs/amazon_test_df_kw_prompted.csv', index=False)

## combine reviews and summaries yelp
get_rev_summ_pairs(yelp_train_df).to_csv(yelp_df + '/sum_pairs/yelp_train_df_kw_prompted.csv', index=False)
get_rev_summ_pairs(yelp_valid_df).to_csv(yelp_df + '/sum_pairs/yelp_valid_df_kw_prompted.csv', index=False)
get_rev_summ_pairs(yelp_test_df).to_csv(yelp_df + '/sum_pairs/yelp_test_df_kw_prompted.csv', index=False)
