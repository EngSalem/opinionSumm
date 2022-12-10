import pandas as pd
import swifter

def replace_mask(reviews):
    if 'Denoise the following reviews:' in reviews:
        return reviews.replace('<mask>','<extra_id_1>')
    else:
        return reviews


df_prompts_train='~/PhD/datasets/yelp_dataset/prompted_yelp_train_gt_20.csv'
df_prompt_valid = '~/PhD/datasets/yelp_dataset/prompted_yelp_valid_gt_20.csv'

df_train=pd.read_csv(df_prompts_train)
df_valid=pd.read_csv(df_prompt_valid)

df_train['input'] = df_train.swifter.apply(lambda row: replace_mask(row['input']), axis=1)
df_valid['input'] = df_valid.swifter.apply(lambda row: replace_mask(row['input']), axis=1)

df_train.to_csv('~/PhD/datasets/yelp_dataset/prompted_yelp_train_t5_gt_20.csv', index=False)
df_valid.to_csv('~/PhD/datasets/yelp_dataset/prompted_yelp_valid_t5_gt_20.csv', index=False)