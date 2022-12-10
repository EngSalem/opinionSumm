import pandas as pd
import os
from sklearn.model_selection import train_test_split

base_dir = '../../courseMirrorSummarization/CourseMirror_data'

df_summ_ref_cs0445 = pd.read_csv('../courseMirrorSummarization/CourseMirror_data/sum_pairs/cs0445_concept_prompted.csv')
df_summ_ref_ie256 = pd.read_csv('../courseMirrorSummarization/CourseMirror_data/sum_pairs/ie256_concept_prompted.csv')
df_summ_ref_ie256_2016 = pd.read_csv('../courseMirrorSummarization/CourseMirror_data/sum_pairs/ie256_2016_concept_prompted.csv')
df_summ_ref_engr = pd.read_csv('../courseMirrorSummarization/CourseMirror_data/sum_pairs/engr_concept_prompted.csv')

## create multiple trains multiple tests
df_summ_ref_cs0445_train, df_summ_ref_cs0445_valid = train_test_split(df_summ_ref_cs0445, test_size=0.1)
df_summ_ref_ie256_train, df_summ_ref_ie256_valid = train_test_split(df_summ_ref_ie256, test_size=0.1)
df_summ_ref_ie256_2016_train, df_summ_ref_ie256_2016_valid = train_test_split(df_summ_ref_ie256_2016, test_size=0.1)
df_summ_ref_engr_train, df_summ_ref_engr_valid = train_test_split(df_summ_ref_engr, test_size=0.1)

train_except_engr = pd.concat([df_summ_ref_cs0445_train, df_summ_ref_ie256_train, df_summ_ref_ie256_2016_train])
valid_except_engr = pd.concat([df_summ_ref_cs0445_valid, df_summ_ref_ie256_valid, df_summ_ref_ie256_2016_valid])

train_except_cs = pd.concat([df_summ_ref_ie256_train, df_summ_ref_ie256_2016_train, df_summ_ref_engr_train])
valid_except_cs = pd.concat([df_summ_ref_ie256_valid, df_summ_ref_ie256_2016_valid, df_summ_ref_engr_valid])

train_except_ie256 = pd.concat([df_summ_ref_cs0445_train, df_summ_ref_ie256_2016_train, df_summ_ref_engr_train])
valid_except_ie256 = pd.concat([df_summ_ref_cs0445_valid, df_summ_ref_ie256_2016_valid, df_summ_ref_engr_valid])

train_except_ie256_2016 = pd.concat([df_summ_ref_cs0445_train, df_summ_ref_ie256_train, df_summ_ref_engr_train])
valid_except_ie256_2016 = pd.concat([df_summ_ref_cs0445_valid, df_summ_ref_ie256_valid, df_summ_ref_engr_valid])

print('Dumping files ...')
train_except_engr.to_csv('../courseMirrorSummarization/CourseMirror_data/sum_pairs/train_except_engr_concept_prompted.csv', index=False)
valid_except_engr.to_csv('../courseMirrorSummarization/CourseMirror_data/sum_pairs/valid_except_engr_concept_prompted.csv', index=False)

train_except_cs.to_csv('../courseMirrorSummarization/CourseMirror_data/sum_pairs/train_except_cs_concept_prompted.csv', index=False)
valid_except_cs.to_csv('../courseMirrorSummarization/CourseMirror_data/sum_pairs/valid_except_cs_concept_prompted.csv', index=False)

train_except_ie256.to_csv('../courseMirrorSummarization/CourseMirror_data/sum_pairs/train_except_ie256_concept_prompted.csv', index=False)
valid_except_ie256.to_csv('../courseMirrorSummarization/CourseMirror_data/sum_pairs/valid_except_ie256_concept_prompted.csv', index=False)

train_except_ie256_2016.to_csv('../courseMirrorSummarization/CourseMirror_data/sum_pairs/train_except_ie256_2016_concept_prompted.csv', index=False)
valid_except_ie256_2016.to_csv('../courseMirrorSummarization/CourseMirror_data/sum_pairs/valid_except_ie256_2016_concept_prompted.csv', index=False)
