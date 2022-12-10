import pandas as pd
import argparse
import swifter
from summ_eval.bert_score_metric import BertScoreMetric
import nltk

## get rouge

bert_score = BertScoreMetric()

def get_bert_score(summary, ref_summaries):
    bert_score_dict = bert_score.evaluate_example(summary, ref_summaries)
    return bert_score_dict['bert_score_f1']

my_parser = argparse.ArgumentParser()
my_parser.add_argument("-reference_summaries", type=str)

my_parser.add_argument("-out_summaries", type=str)
args = my_parser.parse_args()

df_pseudo_summaries = pd.read_csv(args.reference_summaries)

#df_valid['tokenized_summary'] = df_valid.swifter.apply(lambda row: '.\n'.join(nltk.sent_tokenize(row['summary'])), axis=1)
#df_pseudo_summaries['tokenized_summary'] = df_pseudo_summaries.swifter.apply(lambda row: '.\n'.join(nltk.sent_tokenize(row['summary'])), axis=1)

df_pseudo_summaries['bert-score'] = df_pseudo_summaries.apply(lambda row: get_bert_score(row['review'],row['summary']), axis=1)

df_pseudo_summaries.to_csv(args.out_summaries, index=False)

