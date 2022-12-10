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

amazon_df = '../../../FewSum/artifacts/amazon/gold_summs'
yelp_df = '../../../FewSum/artifacts/yelp/gold_summs'


df_amazon_train = pd.read_csv(amazon_df+'/sum_pairs/amazon_train_df.csv')
df_amazon_train['summ_length'] = df_amazon_train.apply(lambda row: get_tokenize_lengths(row['summary']), axis=1)

df_yelp_train = pd.read_csv(yelp_df+'/sum_pairs/yelp_train_df.csv')
df_yelp_train['summ_length'] = df_yelp_train.apply(lambda row: get_tokenize_lengths(row['summary']), axis=1)

sns.distplot(df_amazon_train['summ_length'].tolist(), label='Amazon')
sns.distplot(df_yelp_train['summ_length'].tolist(), label='Yelp')

plt.xlabel('Summary Length')
plt.ylabel('Length Distribution')
plt.legend()
plt.show()
