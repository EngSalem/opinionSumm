import pandas as pd
import swifter
from sklearn.model_selection import train_test_split

extractive_prompt = 'Extract the 2 key sentences from the following reviews:'
paraphrased_prompt = 'Could you rephrase the following reviews?' \
                     'review:'


extractive_data = '~/PhD/datasets/yelp_dataset/yelp_unannotated_bank_with_extractive_summ_20k.csv'
paraphrased_data = '~/PhD/datasets/yelp_dataset/yelp_bank_paraphrased_20000.csv'

df_paraphrased = pd.read_csv(paraphrased_data).head(20000)
df_extractive = pd.read_csv(extractive_data)

df_paraphrased['prompted_paraphrased'] = df_paraphrased.swifter.apply(lambda row: '\n'.join([paraphrased_prompt,row['review']]), axis=1)
df_extractive['prompted_extractive'] = df_extractive.swifter.apply(lambda row: ' '.join([extractive_prompt,row['review']]), axis=1)

prompted_data = pd.DataFrame(data={'review':df_extractive['prompted_extractive'].tolist()+df_paraphrased['prompted_paraphrased'].tolist(),
                   'summary':df_extractive['summary'].tolist()+df_paraphrased['paraphrased_review'].tolist()}).sample(frac=1.)

train_prompted, valid_prompted = train_test_split(prompted_data)

train_prompted.to_csv('~/PhD/datasets/yelp_dataset/prompted_yelp_train.csv', index=False)
valid_prompted.to_csv('~/PhD/datasets/yelp_dataset/prompted_yelp_valid.csv', index=False)