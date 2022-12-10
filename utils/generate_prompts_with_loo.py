import pandas as pd
import swifter
from sklearn.model_selection import train_test_split

extractive_prompt = 'Extract the key sentences from the following reviews:'
loo_prompt = 'Given the following reviews, generate a similar review: ' \
                     'reviews:'
dapt_prompt = 'Denoise the following reviews:'

loo_data = '~/PhD/datasets/yelp_dataset/yelp_loo_rouge_gt_20.csv'
extractive_data = '~/PhD/datasets/yelp_dataset/yelp_extractive_gt_20.csv'
dapt_data = '~/PhD/datasets/yelp_dataset/yelp_loo_with_dapt_gt_20.csv'

df_loo = pd.read_csv(loo_data)
df_extractive = pd.read_csv(extractive_data)
df_dapt = pd.read_csv(dapt_data)

df_loo['prompted_reviews'] = df_loo.swifter.apply(lambda row: '\n'.join([loo_prompt,row['reviews']]), axis=1)
df_extractive['prompted_reviews'] = df_extractive.swifter.apply(lambda row: ' '.join([extractive_prompt,row['reviews']]), axis=1)
df_dapt['prompted_reviews'] = df_dapt.swifter.apply(lambda row: ' '.join([dapt_prompt,row['dapt_reviews']]), axis=1)



prompted_data = pd.DataFrame(data={'input':df_extractive['prompted_reviews'].tolist()+df_loo['prompted_reviews'].tolist() +df_dapt['prompted_reviews'].tolist(),
                   'output':df_extractive['extracted_summaries'].tolist()+df_loo['summary'].tolist()+df_dapt['reviews'].tolist()}).sample(frac=1.)

train_prompted, valid_prompted = train_test_split(prompted_data,test_size=0.05)

train_prompted.to_csv('~/PhD/datasets/yelp_dataset/prompted_yelp_train_gt_20.csv', index=False)
valid_prompted.to_csv('~/PhD/datasets/yelp_dataset/prompted_yelp_valid_gt_20.csv', index=False)