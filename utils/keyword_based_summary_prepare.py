import aspect_extractor as key_extractor
import pandas as pd
import swifter

def get_keyword_planned_prompts(review, summary):
    keywords = key_extractor.get_final_kw_list(summary)

    return '[ASPECTCHAIN]'+ ' | '.join(keywords) +' [SUMMARY] '+summary

df_train = pd.read_csv('../FewSumm/artifacts/amazon/gold_summs/sum_pairs/amazon_train_df.csv')
df_valid = pd.read_csv('../FewSumm/artifacts/amazon/gold_summs/sum_pairs/amazon_train_df.csv')

df_train['aspect_summary_chain'] = df_train.apply(lambda row: get_keyword_planned_prompts(row['review'], row['summary']), axis=1)
df_train['aspect_summary_chain'] = df_train.apply(lambda row: get_keyword_planned_prompts(row['review'], row['summary']), axis=1)

df_train.to_csv('../FewSumm/artifacts/amazon/gold_summs/sum_pairs/amazon_train_df_aspect_sentiment_chain.csv', index=False)
df_train.to_csv('../FewSumm/artifacts/amazon/gold_summs/sum_pairs/amazon_valid_df_aspect_sentiment_chain.csv', index=False)
