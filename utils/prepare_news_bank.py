import pandas as pd
from datasets import load_dataset

datasets = ['multi_news','xsum']

def get_article_summ(dataset, article_col, summary_col):
    articles, summaries = [], []
    for article, summary in zip(dataset['train'][article_col], dataset['train'][summary_col]):
        articles.append(' '.join(article.strip().split()))
        summaries.append(' '.join(summary.strip().split()))

    return pd.DataFrame(data={'article':articles, 'summary':summaries})


sets= []
for dataset in datasets:
    data = load_dataset(dataset)
    ## prepare multi news
    sets.append(get_article_summ(data, 'document','summary'))

sets.append(pd.read_csv('~/PhD/datasets/cnn-dailymail/cnn-dailymail_train.csv'))

df_news = pd.concat(sets)
df_news.to_csv('~/PhD/datasets/news_summaries.csv')


