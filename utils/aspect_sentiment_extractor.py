import pandas as pd
import spacy
import aspect_extractor as ae
import swifter
import stanza
import argparse
from transformers import pipeline


parser = argparse.ArgumentParser()
parser.add_argument('-file', type=str)
parser.add_argument('-col',type=str)
parser.add_argument('-summary', type=boolean)
parser.add_argument('-out_file', type=str)
args = parser.parse_args()


classifier = pipeline("zero-shot-classification",
                              model="facebook/bart-large-mnli")
candidate_labels = ['positive', 'negative', 'neutral']


stanza.download('en')
nlp = stanza.Pipeline('en')


def get_aspect_sentiment_candidates(reviews):
    print(type(reviews))
    doc = nlp(reviews)
    ## iterate over sentences
    candidates = []
    for sent in doc.sentences:
        ## get dependencies
        for i in range(len(sent.dependencies)):
            if (sent.dependencies[i][-1].upos == 'ADJ' and sent.dependencies[i][1] == 'amod' and sent.dependencies[i][
                0].upos == 'NOUN'):
                candidates.append((sent.dependencies[i][-1].text, sent.dependencies[i][0].text))

            elif (sent.dependencies[i][-1].upos == 'NOUN' and sent.dependencies[i][1] == 'nsubj' and
                  sent.dependencies[i][0].upos == 'ADJ'):
                candidates.append((sent.dependencies[i][0].text, sent.dependencies[i][-1].text))

    return candidates


def filter_candidates(reviews, dep_list):
    ## get keyword list
    kw_list = ae.get_final_kw_list(reviews)
    final_pairs = []
    for dep in dep_list:
        if dep[1] in kw_list and filter_aspect_sentiment((' '.join([dep[0], dep[1]]))):
            final_pairs.append(','.join([dep[0], dep[1]]))
    return final_pairs


def get_aspect_Sentiment_prompts(reviews):
    dep_candidates = get_aspect_sentiment_candidates(reviews)
    if len(dep_candidates) > 0:
        final_dep_list = filter_candidates(reviews, dep_candidates)
    return ' | '.join(final_dep_list)


def get_aspect_sentiment_pairs_only(reviews, summary=False):
    dep_candidates = get_aspect_sentiment_candidates(reviews)

    if len(dep_candidates) > 0 and not summary:
        final_dep_list = filter_candidates(reviews, dep_candidates)
        return final_dep_list

    elif len(dep_candidates) > 0 and summary:
        final_dep_list = [','.join([dep[0], dep[1]]) for dep in dep_candidates if filter_aspect_sentiment(dep)]
        return final_dep_list
    return dep_candidates


def filter_aspect_sentiment(aspect_sentiment):
    labels = classifier(aspect_sentiment, candidate_labels, device=0)
    if (labels['labels'][0] == 'positive' or labels['labels'][0] == 'negative') and labels['scores'][0] >= 0.8:
        return True
    else:
        return False


df_data = pd.read_csv(args.file)
df_data[args.file+'_aspect_sentiment'] = df_data.apply(lambda row: get_aspect_sentiment_pairs_only(row[args.col], args.summary), axis=1)

df_data.to_csv(args.out_file, index=False)


