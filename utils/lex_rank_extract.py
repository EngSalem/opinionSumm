from lexrank import LexRank
from lexrank.mappings.stopwords import STOPWORDS
from path import Path
from nltk.tokenize import sent_tokenize
import argparse

my_parser = argparse.ArgumentParser()
my_parser.add_argument("-reviews", type=str)
my_parser.add_argument("-summaries", type=str)
args = my_parser.parse_args()

reviews  = [ sent_tokenize(review) for review in open(args.reviews, 'r').readlines()]

lxr = LexRank(reviews, stopwords=STOPWORDS['en'])

review = reviews[0]
summary = lxr.get_summary(review, summary_size=2, threshold=.1)
print(summary)


