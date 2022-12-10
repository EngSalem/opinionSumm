import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--review_data', action='store', dest='review_data',
                    help='')
parser.add_argument('--output', action='store', dest='output',
                    help='')

args = parser.parse_args()

review_df = pd.read_csv(args.review_data)
review_df['processed_reviews'] = review_df.apply(lambda row: ' '.join(row['reviews'].split('\n')), axis=1)

fw = open(args.output, 'w')
for rev in review_df['processed_reviews'].tolist():
    fw.write(rev+'\n')
fw.close()