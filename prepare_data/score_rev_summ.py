from rouge_score import rouge_scorer
import swifter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
ROUGE_TH=0.2

df_loo = None

try:
    df_loo = pd.read_csv('~/PhD/datasets/yelp_dataset/yelp_leave_one_out_with_rouge1.csv')

except:
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    df_loo = pd.read_csv('~/PhD/datasets/yelp_dataset/yelp_leave_one_out.csv')
    df_loo.dropna(inplace=True)
    df_loo['rouge-1'] = df_loo.swifter.apply(lambda row: scorer.score(row['reviews'], row['summary'])['rouge1'].fmeasure, axis=1)
    df_loo.to_csv('~/PhD/datasets/yelp_dataset/yelp_leave_one_out_with_rouge1.csv', index=False)

#sns.histplot(data=df_loo, x='rouge-1')

#df_loo = df_loo.sample(frac=.1)
#df_loo['rouge-1'].hist()
#plt.show()

df_loo[df_loo['rouge-1'] >=0.2].to_csv('~/PhD/datasets/yelp_dataset/yelp_loo_rouge_gt_20.csv', index=False)