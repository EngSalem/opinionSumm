import os
import pandas as pd

chunksDir = '/home/mohamed/PhD/datasets/yelp_dataset/leave_one_out/'
outDir = '/home/mohamed/PhD/datasets/yelp_dataset/'

dfs = []
for file in os.listdir(chunksDir):
    df = pd.read_csv(os.path.join(chunksDir,file))
    df.dropna(inplace=True)
    ## normalize
    df['reviews'] = [' '.join(review.split('\n')) for review in df['reviews'].tolist()]
    df['summary'] = [' '.join(review.split('\n')) for review in df['summary'].tolist()]
    ## add df
    dfs.append(df)

pd.concat(dfs).to_csv(os.path.join(outDir,'yelp_leave_one_out.csv'), index=False)


