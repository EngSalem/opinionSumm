import pandas as pd
import ast

def construct_summary_with_aspect_sentiment_chain( summary, aspect_sentiment):
    aspect_sentiment_chain = ' | '.join(ast.literal_eval(aspect_sentiment))

    return ' '.join(['[ASPECTCHAIN]', aspect_sentiment_chain, '[SUMMARY]', summary])



train_df = pd.read_csv('../FewSumm/artifacts/yelp/summ_aspects_sentiments/generated_as_pairs_yelp_train.csv')
valid_df = pd.read_csv('../FewSumm/artifacts/yelp/summ_aspects_sentiments/generated_as_pairs_yelp_valid.csv')

train_df['chain_summary'] = train_df.apply(lambda row: construct_summary_with_aspect_sentiment_chain(row['summary'],
                                                                                                     row['aspect_sentiment_for_summary']), axis=1)

valid_df['chain_summary'] = valid_df.apply(lambda row: construct_summary_with_aspect_sentiment_chain(row['summary'],
                                                                                                     row['aspect_sentiment_for_summary']), axis=1)

train_df.to_csv('../FewSumm/artifacts/yelp/summ_aspects_sentiments/train_df_aspect_sentiment_chain.csv', index=False)
valid_df.to_csv('../FewSumm/artifacts/yelp/summ_aspects_sentiments/valid_df_aspect_sentiment_chain.csv', index=False)
