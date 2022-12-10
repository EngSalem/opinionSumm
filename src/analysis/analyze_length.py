from nltk.tokenize import word_tokenize
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('darkgrid')

def get_tokenize_lengths(text):
    '''
    :param text: input text
    :return: length of tokenized text
    '''
    return len(word_tokenize(text))


df_cs_valid = pd.read_csv('../../courseMirrorSummarization/CourseMirror_data/sum_pairs/valid_except_cs.csv')
df_cs_valid['summary_len'] = df_cs_valid.apply(lambda row: get_tokenize_lengths(row['summary']), axis=1)

df_ie256_valid = pd.read_csv('../../courseMirrorSummarization/CourseMirror_data/sum_pairs/valid_except_ie256.csv')
df_ie256_valid['summary_len'] = df_ie256_valid.apply(lambda row: get_tokenize_lengths(row['summary']), axis=1)

df_ie256_2016_valid = pd.read_csv('../../courseMirrorSummarization/CourseMirror_data/sum_pairs/valid_except_ie256_2016.csv')
df_ie256_2016_valid['summary_len'] = df_ie256_2016_valid.apply(lambda row: get_tokenize_lengths(row['summary']), axis=1)

df_engr_valid = pd.read_csv('../../courseMirrorSummarization/CourseMirror_data/sum_pairs/valid_except_engr.csv')
df_engr_valid['summary_len'] = df_engr_valid.apply(lambda row: get_tokenize_lengths(row['summary']), axis=1)

sns.distplot(df_cs_valid['summary_len'].tolist(), label='Except CS')
sns.distplot(df_ie256_valid['summary_len'].tolist(), label='Except IE256')
sns.distplot(df_ie256_2016_valid['summary_len'].tolist(), label='Except IE256 2016')
sns.distplot(df_engr_valid['summary_len'].tolist(), label='ENGR')

plt.xlabel('summary Lengths')
plt.ylabel('Length Density')
plt.show()