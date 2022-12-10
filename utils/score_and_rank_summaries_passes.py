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
             candidate = ' '.join(' '.join(pairs[i:j]).replace("\'","").replace('\'','').split(' ')).lower()
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
       return recall
    else:
       return 0


def score_and_decide(reference, passes):
    scores = {'pass_k0':0, 'pass_k1':0, 'pass_k2':0, 'pass_k3':0}


    for pass_count in range(4):
        scores['pass_k'+str(pass_count)] = get_recall(reference, passes[pass_count])


    if scores[max(scores, key=scores.get)] > scores['pass_k0']:
       return  max(scores, key=scores.get)

    else:
       return 'pass_k0'




def get_final_summaries(df):
    fial_decisions = []
    for reference, generated_summ_as, generated_summ_as_k1, generated_summ_as_k2, generated_summ_as_k3,\
        generated_summ, generated_summ_k1, generated_summ_k2, generated_summ_k3 in zip(df['review_aspect_sentiment'].tolist(),
                                                                           df['summary_aspect_sentiment'].tolist(),
                                                                          df['k1_aspect_sentiment'].tolist(),
                                                                          df['k2_aspect_sentiment'].tolist(),
                                                                          df['k3_aspect_sentiment'].tolist(),
                                                                          df['generated_summary'].tolist(),
                                                                          df['generated_summary_k1'].tolist(),
                                                                          df['generated_summary_k2'].tolist(),
                                                                          df['generated_summary_k3'].tolist()):

        reference = filter_aspect_sentiment(reference)
        generated_summ_as = filter_aspect_sentiment(generated_summ_as)
        generated_summ_as_k1 = filter_aspect_sentiment(generated_summ_as_k1)
        generated_summ_as_k2 = filter_aspect_sentiment(generated_summ_as_k2)
        generated_summ_as_k3 = filter_aspect_sentiment(generated_summ_as_k3)

        if len(reference) >0:
           dec = score_and_decide(reference, [generated_summ_as, generated_summ_as_k1, generated_summ_as_k2, generated_summ_as_k3])
        else:
           dec = 'pass_k0'

        if dec == 'pass_k0':
           fial_decisions.append(generated_summ)
        elif dec == 'pass_k1':
           fial_decisions.append(generated_summ_k1)
        elif dec == 'pass_k2':
           fial_decisions.append(generated_summ_k2)
        elif dec == 'pass_k3':
            fial_decisions.append(generated_summ_k3)
        print(dec)
    return fial_decisions





df_summaries = pd.read_csv('../courseMirror/src/summarization/amazon_test_df_aspect_sentiment_multi_pertubed.csv')


df_summaries['generated_summaries'] = get_final_summaries(df_summaries)

df_summaries.to_csv('../courseMirror/src/summarization/amazon_test_df_aspect_sentiment_multi_pertubed_auto_dec.csv', index=False)