from nltk.tokenize import sent_tokenize
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import swifter


sns.set_style('darkgrid')

df_yelp = pd.read_csv('../../../FewSumm/artifacts/yelp/gold_summs/sum_pairs/yelp_train_df.csv')
df_yelp['sent_lens'] = df_yelp.swifter.apply(lambda row: len(sent_tokenize(row['review'])), axis=1)

sns.histplot(data=df_yelp, x='sent_lens')

plt.show()