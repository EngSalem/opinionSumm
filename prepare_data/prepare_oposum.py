import os
import pandas as pd

categories = ['bags_and_cases','bluetooth','boots','keyboards','tv','vacuums']
summ_dir = '../oposum/data/gold/summaries/'
review_dir = '../oposum-train/'

def get_id2reviews(file):
    ids,reviews = [], []
    text = open(file,'r').read()
    texts= text.split('\n\n')
    for text in texts:
        try:
          ids.append(text.split('\n')[0].split()[0])
          reviews.append(' '.join(text.split('\n')[1:]))
        except:
          continue

    return pd.DataFrame(data={'ids':ids,'reviews':reviews})



dfs = []
for category in categories:
    dfs.append(get_id2reviews(os.path.join(review_dir,category+'.trn')))
pd.concat(dfs).to_csv(os.path.join(review_dir,'review_df.csv'), index=False)