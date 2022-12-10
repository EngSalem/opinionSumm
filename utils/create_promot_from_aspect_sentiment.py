import pandas as pd
from ast import literal_eval

def get_prompt(pairs, reviews):
    pairs = literal_eval(pairs)
    return ' | '.join(pairs)+' | reviews: '+ reviews

df_train = pd.read_csv('../FewSumm/artifacts/amazon/gold_summs/sum_pairs/amazon_train_df_with_review_aspect_sentiment_kw_2.csv')
df_valid = pd.read_csv('../FewSumm/artifacts/amazon/gold_summs/sum_pairs/amazon_valid_df_with_review_aspect_sentiment_kw_2.csv')
df_test = pd.read_csv('../FewSumm/artifacts/amazon/gold_summs/sum_pairs/amazon_test_df_with_review_aspect_sentiment_kw_2.csv')

df_train['prompted_reviews'] = df_train.apply(lambda row: get_prompt(row['aspect_sentiment'], row['review']), axis=1)
df_valid['prompted_reviews'] = df_valid.apply(lambda row: get_prompt(row['aspect_sentiment'], row['review']), axis=1)
df_test['prompted_reviews'] = df_test.apply(lambda row: get_prompt(row['aspect_sentiment'], row['review']), axis=1)

df_train.to_csv('../FewSumm/artifacts/amazon/gold_summs/sum_pairs/amazon_train_df_review_aspect_sentiment_prompted_kw_2.csv', index=False)
df_valid.to_csv('../FewSumm/artifacts/amazon/gold_summs/sum_pairs/amazon_valid_df_review_aspect_sentiment_prompted_kw_2.csv', index=False)
df_test.to_csv('../FewSumm/artifacts/amazon/gold_summs/sum_pairs/amazon_test_df_review_aspect_sentiment_prompted_kw_2.csv', index=False)