import pandas as pd
import spacy
# import aspect_extractor as ae
import swifter
import stanza
import argparse
from transformers import pipeline

stanza.download('en')
nlp = stanza.Pipeline('en')


## Rules for aspect sentiment pairs: R1:  (NN) -> nsubj -> (adj) | R2: (ADJ) -> amod (NOUN)
## Rules for merging aspect words: R3: (NN) ->nn -> (NN) | R4:
## Rules for merging sentiment words: R5: (ADJ)-> conj (ADJ)
def get_dependencies(text):
    doc = nlp(text)

    final_candidates = []

    ## iterate over sentences
    for sent in doc.sentences:

        candidates = []
        sentiment_word_pairs = []
        aspect_word_pairs = []

        for i, dependency in enumerate(sent.dependencies):
            ############ rules for aspect-sentiment pairs
            ## apply rule 1
            if ((sent.dependencies[i][-1].upos == 'NOUN' or sent.dependencies[i][-1].upos == 'PROPN') and
                    sent.dependencies[i][0].upos == 'ADJ' and \
                    sent.dependencies[i][1] == 'nsubj'):
                candidates.append((sent.dependencies[i][0].text, sent.dependencies[i][-1].text))
            ## apply rule 2
            elif sent.dependencies[i][-1].upos == 'ADJ' and (
                    sent.dependencies[i][0].upos == 'NOUN' or sent.dependencies[i][0].upos == 'PROPN') and \
                    sent.dependencies[i][1] == 'amod':
                candidates.append((sent.dependencies[i][-1].text, sent.dependencies[i][0].text))

            ########### rules for aspect-words #########
            ## appply rule 3

            elif (sent.dependencies[i][-1].upos == 'PROPN' or sent.dependencies[i][-1].upos == 'NOUN') and (
                    sent.dependencies[i][0].upos == 'PROPN' or sent.dependencies[i][-1].upos == 'NOUN') and \
                    sent.dependencies[i][1] == 'compound':
                aspect_word_pairs.append((sent.dependencies[i][-1].text, sent.dependencies[i][0].text))

            ## apply rule 4
            elif (sent.dependencies[i][-1].upos == 'PROPN' or sent.dependencies[i][-1].upos == 'NOUN') and (
                    sent.dependencies[i][0].upos == 'PROPN' or sent.dependencies[i][-1].upos == 'NOUN') and \
                    sent.dependencies[i][1] == 'conj':
                aspect_word_pairs.append(
                    (sent.dependencies[i][-1].text, sent.dependencies[i][0].text))

            ####### rules for multiple sentiment words
            ## apply rule 5

            elif sent.dependencies[i][-1].upos == 'ADJ' and sent.dependencies[i][0].upos == 'ADJ' and \
                    sent.dependencies[i][1] == 'conj':
                sentiment_word_pairs.append(
                    (sent.dependencies[i][-1].text, sent.dependencies[i][0].text))

        #####  merge the aspect words and expand

        for candidate in candidates:
            added = False

            curr_candidates = []
            for aspect_pair in aspect_word_pairs:
                ## adjust aspect pairs
                print(aspect_pair)
                if candidate[1] in aspect_pair:
                    added = True
                    final_candidates.append((candidate[0], aspect_pair[0] + ' ' + aspect_pair[1]))
                    curr_candidates.append((candidate[0], aspect_pair[0] + ' ' + aspect_pair[1]))

            if not added:
                final_candidates.append(candidate)
                curr_candidates.append(candidate)

            for sentiment_pair in sentiment_word_pairs:
                ## expand sentiment list
                for curr_candidate in curr_candidates:
                    if curr_candidate[0] in sentiment_pair:
                        final_candidates.extend([(sentiment_pair[0], curr_candidate[-1]), (sentiment_pair[1], curr_candidate[-1])])

    return set(final_candidates)


candidates = get_dependencies('''
I had an almost identical steamer from Black & Decker purchased around 1985. The lid on that one finally broke and I've been looking for one like it for 2 years. Nice to see a good design hasn't been changed. Love the steamer!
I bought this one as a replacement for an old one I had. We cook with it every day and love it! It is the perfect size for our family.
''')

print(candidates)