import json
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--json_in', action='store', dest='json_in',
                    help='')
parser.add_argument('--dataframe_out', action='store', dest='dataframe_out',
                    help='')

args = parser.parse_args()

yelp_reviews = args.json_in

reviews_dicts = []
for line in open(yelp_reviews,'r').readlines():
    reviews_dicts.append(json.loads(line.strip()))


yelp_df = pd.DataFrame(reviews_dicts)
yelp_df.to_csv(args.dataframe_out, index=False)