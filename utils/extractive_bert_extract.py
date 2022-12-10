from summarizer import Summarizer
import pandas as pd
import swifter
import argparse

my_parser = argparse.ArgumentParser()
my_parser.add_argument("-reviews", type=str)
my_parser.add_argument("-summaries", type=str)
my_parser.add_argument("-num_sent", type=int)
args = my_parser.parse_args()

reviews  = pd.read_csv(args.reviews)['reviews'].tolist()

model = Summarizer()
#result = model(reviews[0], ratio=0.2)  # Specified with ratio
#summary = model(reviews[0], num_sentences=args.num_sent)

rev_df = pd.DataFrame(data={'reviews': reviews})
rev_df['extracted_summaries'] = rev_df.swifter.apply(lambda row: model(row['reviews'], num_sentences=args.num_sent), axis=1)

rev_df.to_csv(args.summaries, index=False)