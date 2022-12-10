import pandas as pd
import swifter

summ_prompt = 'TLDR; Can you summarize the following reviews? ' \
              'review:'

train='../FewSumm/artifacts/amazon/gold_summs/sum_pairs/amazon_train_df.csv'
valid='../FewSumm/artifacts/amazon/gold_summs/sum_pairs/amazon_valid_df.csv'
test='../FewSumm/artifacts/amazon/gold_summs/sum_pairs/amazon_test_df.csv'

df_train = pd.read_csv(train)
df_valid = pd.read_csv(valid)
df_test = pd.read_csv(test)

df_train['prompted_reviews'] = df_train.swifter.apply(lambda row: ' '.join([summ_prompt,row['review']]), axis=1)
df_valid['prompted_reviews'] = df_valid.swifter.apply(lambda row: ' '.join([summ_prompt,row['review']]), axis=1)
df_test['prompted_reviews'] = df_test.swifter.apply(lambda row: ' '.join([summ_prompt,row['review']]), axis=1)

df_train.to_csv('../FewSumm/artifacts/amazon/gold_summs/sum_pairs/amazon_train_prompted_df.csv', index=False)
df_valid.to_csv('../FewSumm/artifacts/amazon/gold_summs/sum_pairs/amazon_valid_prompted_df.csv', index=False)
df_test.to_csv('../FewSumm/artifacts/amazon/gold_summs/sum_pairs/amazon_test_prompted_df.csv', index=False)
