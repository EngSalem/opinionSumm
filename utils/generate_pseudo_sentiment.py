from transformers import pipeline
from nltk.tokenize import sent_tokenize
from collections import Counter
import swifter
import pandas as pd
import argparse

my_parser = argparse.ArgumentParser()
my_parser.add_argument("-reviews", type=str)
my_parser.add_argument("-sentiment", type=str)
my_parser.add_argument("-device", type=int)
args = my_parser.parse_args()

classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

candidate_labels = ['positive', 'negative', 'neutral']

def general_sentiment(review):
    sentences = sent_tokenize(review)
    labels = classifier(sentences, candidate_labels, device=args.device)
    predictions = []
    for l in labels:
        predictions.append(l['labels'][0])
    return pd.DataFrame(data={'preds':predictions})['preds'].mode()[0]

reviews  = pd.read_csv(args.reviews)['reviews'].tolist()
df_reviews = pd.DataFrame(data={'review':reviews})
df_reviews['sentiment'] = df_reviews.swifter.apply(lambda row: general_sentiment(row['review']), axis=1)

df_reviews.to_csv(args.sentiment, index=False)