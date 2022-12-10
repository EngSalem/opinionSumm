import pandas as pd
#from sentence_transformers import SentenceTransformer
#from sklearn.metrics.pairwise import cosine_similarity

#model = SentenceTransformer('all-MiniLM-L6-v2')

def filter_aspect_sentiment(candidates):
    candidates = candidates.strip(']').strip('[').strip("\'").strip("\"")
    if len(candidates) >  0:
       pairs = candidates.split(',')
       i=0
       j=2
       final_candidates = []
       while j< len(pairs):
             candidate = ' '.join(' '.join(pairs[i:j]).replace("\'","").replace('\'','').split(' '))
             if candidate[0]==' ':
                final_candidates.append(candidate[1:])
             else:
                final_candidates.append(candidate)
             i+=2
             j+=2

       return final_candidates

    else:
       return []

def get_recall(reference_list, summary_list):

    if len(summary_list) > 0:
       recall =  len(list(set(reference_list) & set(summary_list)))/len(set(reference_list))
       print(recall)
       return recall
    else:
       return 0


def score_and_decide(reference, beams):
    scores = {'beam1':0, 'beam2':0, 'beam3':0, 'beam4':0, 'beam5':0}


    for beam_cout in range(1,6):

        scores['beam'+str(beam_cout)] = get_recall(reference, beams[beam_cout-1])


    if scores[max(scores, key=scores.get)] > scores['beam4']:
       return  max(scores, key=scores.get)

    else:
       return 'beam4'




def get_final_summaries(df):
    fial_decisions = []
    for reference, beam1_as, beam2_as, beam3_as, beam4_as, beam5_as, beam1, beam2, beam3, beam4, beam5 in zip(df['aspect_sentiment_pairs_reviews'].tolist(),
                                                                          df['beam1_aspect_sentiment'].tolist(),
                                                                           df['beam2_aspect_sentiment'].tolist(),
                                                                           df['beam3_aspect_sentiment'].tolist(),
                                                                           df['beam4_aspect_sentiment'].tolist(),
                                                                           df['beam5_aspect_sentiment'].tolist(),
                                                                           df['generated_summary_beam1'].tolist(),
                                                                           df['generated_summary_beam2'].tolist(),
                                                                           df['generated_summary_beam3'].tolist(),
                                                                           df['generated_summary_beam4'].tolist(),
                                                                           df['generated_summary_beam5'].tolist() ):

        reference = filter_aspect_sentiment(reference)
        beam1_as = filter_aspect_sentiment(beam1_as)
        beam2_as = filter_aspect_sentiment(beam2_as)
        beam3_as = filter_aspect_sentiment(beam3_as)
        beam4_as = filter_aspect_sentiment(beam4_as)
        beam5_as = filter_aspect_sentiment(beam5_as)

        if len(reference) >0:
           dec = score_and_decide(reference, [beam1_as, beam2_as, beam3_as, beam4_as, beam5_as])
        else:
           dec = 'beam4'
        if dec == 'beam1':
           fial_decisions.append(beam1)
        elif dec == 'beam2':
           fial_decisions.append(beam2)
        elif dec == 'beam3':
           fial_decisions.append(beam3)
        elif dec == 'beam4':
           fial_decisions.append(beam4)
        elif dec == 'beam5':
           fial_decisions.append(beam5)

        print(dec)
    return fial_decisions





df_summaries = pd.read_csv('../courseMirror/src/summarization/yelp_test_df_aspect_sentiment.csv')


df_summaries['generated_summaries'] = get_final_summaries(df_summaries)

df_summaries.to_csv('../courseMirror/src/summarization/yelp_test_df_aspect_sentiment.csv_auto_dec.csv', index=False)