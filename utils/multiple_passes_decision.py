import pandas as pd
from ast import literal_eval

df = pd.read_csv('../courseMirror/src/summarization/outputs/ie256_2016_bart_large_multiple_beams_concepts.csv')
df['concepts'] = df['concepts'].apply(lambda concept: literal_eval(concept))


##
def get_concept_score(concepts, summary):
    concept_counter = 0
    for concept in concepts:
        if concept in summary:
            concept_counter + 1
    return concept_counter / len(concepts)


def score_and_decide(concepts, beam1, beam2, beam3, beam4, beam5):
    scores = {'beam1': 0, 'beam2': 0, 'beam3': 0, 'beam4': 0, 'beam5': 0}
    summaries = {'beam1': beam1, 'beam2': beam2, 'beam3': beam3, 'beam4': beam4, 'beam5': beam5}
    if len(concepts) == 0:
        ## no concepts
        return beam4

    else:
        for ix, beam in enumerate([beam1, beam2, beam3, beam4, beam5]):
            try:
                scores['beam' + str(ix + 1)] = get_concept_score(concepts, beam)
            except:
                pass

    print(max(scores, key=scores.get))
    if scores[max(scores, key=scores.get)] > scores['beam4']:

        return summaries[max(scores, key=scores.get)]

    else:
        return summaries['beam4']


df['generated_summary'] = df.apply(lambda row: score_and_decide(row['concepts'], row['generated_summary_beam1'],
                                                                row['generated_summary_beam2'],
                                                                row['generated_summary_beam3'],
                                                                row['generated_summary_beam4'],
                                                                row['generated_summary_beam5']), axis=1)

df.to_csv('../courseMirror/src/summarization/outputs/ie256_2016_bart_large_multiple_beams_auto_dec.csv', index=False)
