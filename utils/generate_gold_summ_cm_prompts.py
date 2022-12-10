import pandas as pd
import swifter
import os

courses = ['cs','engr','ie256','ie256_2016']

summ_prompt = 'TLDR; Can you summarize the following reflections? ' \
              'reflections:'

CMDIR = '../courseMirror/courseMirrorSummarization/CourseMirror_data/sum_pairs'


for course in courses:
    train = os.path.join(CMDIR,'train_except_'+course+'.csv')
    valid = os.path.join(CMDIR,'valid_except_'+course+'.csv')
    if course == 'cs':
       test = os.path.join(CMDIR,course+'0445.csv')
    df_train = pd.read_csv(train)
    df_valid = pd.read_csv(valid)
    df_test = pd.read_csv(test)

    df_train['prompted_reflections'] = df_train.swifter.apply(lambda row: ' '.join([summ_prompt,row['reference']]), axis=1)
    df_valid['prompted_reflections'] = df_valid.swifter.apply(lambda row: ' '.join([summ_prompt,row['reference']]), axis=1)
    df_test['prompted_reflections'] = df_test.swifter.apply(lambda row: ' '.join([summ_prompt,row['reference']]), axis=1)

    df_train.to_csv(os.path.join(CMDIR,'train_except_'+course+'_prompted.csv'), index=False)
    df_valid.to_csv(os.path.join(CMDIR,'valid_except_'+course+'_prompted.csv'), index=False)
    if course == 'cs':
       df_test.to_csv(os.path.join(CMDIR,course+'0445_prompted.csv'), index=False)
