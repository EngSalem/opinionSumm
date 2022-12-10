from spacy.lang.en import English
import spacy
from sentence_transformers import SentenceTransformer
import re
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import igraph as ig
from collections import Counter
##### space model | stop words | sentence encoder

print('------------ load model ---------------------')
nlp = spacy.load('en_core_web_sm')
STOP_WORDS = nlp.Defaults.stop_words
print('------------- load encoder -------------------')
sentenceEncoder = SentenceTransformer('stsb-roberta-large')
#####
print('------- Done loading encoder --------------------')

ENTITIES_TYPES= ['PERSON', 'NORP','FAC','ORG','GPE','LOC','PRODUCT','EVENT','WORK_OF_ART','LAW','LANGUAGE','DATE','TIME',
                                                                  'PERCENT','MONEY','QUANTITY','ORDINAL','CARDINAL']

def get_nounphrases(text):
    doc = nlp(text)
    noun_phrases =[]
    for np in doc.noun_chunks:
        if np.text:
           noun_phrases.append(np.text)
    return noun_phrases

def get_nouns(text):
    doc = nlp(text)
    nouns = []
    for token in doc:
        if token.pos_ == 'NOUN' or token.pos_ == 'PROPN':
           nouns.append(token.text)
           #print(token.text)
    return nouns

def normalize_text(text):
    #text = text.lower()
    words = []
    for word in text.split():
        words.append(re.sub(r'[^a-zA-Z0-9]', '', word))
    text = ' '.join(words)
    return text

def encode_phrase(txt):
    return sentenceEncoder.encode(txt)

def get_similarityMatrix(vecs, phrases, th=0.8):
    full_sims, pivotphrases, phrasesref = [], [], []
    for i in range(len(vecs)):
        current_phrases = phrases.copy()
        sims = cosine_similarity(vecs[i].reshape(1, -1), vecs)
        sims = np.delete(sims, i)
        full_sims.extend(sims)
        pivotphrases.extend([phrases[i]]*len(sims))
        del current_phrases[i]
        phrasesref.extend(current_phrases)
        #full_sims.append(np.mean(sims))
    df= pd.DataFrame.from_dict({'phrase1': phrasesref, 'phrase2': pivotphrases, 'sims': full_sims})
    return df[df['sims'] >= th]

def filter_phrase_list(ph_list):
    filtered_list = []
    for ph in ph_list:
        if ph not in STOP_WORDS and len(ph) > 2:
           filtered_list.append(ph)
    return filtered_list

def get_phrases_freqcouncy(phrases):
    '''
    :param phrases: i/p phrases
    :return: frequency dictionary of phrases
    '''
    freqDict = Counter(phrases)
    return freqDict

def get_keyWords(num_clusters,components,counts):
    clusters, keywords, keywordscounts = [], [], []
    for i in range(num_clusters + 1):
        # print(i)
        words = [v['name'] for v in components.subgraph(idx=i).vs]
        words_counts = [counts[w] for w in words]
        df_words = pd.DataFrame.from_dict({'word': words, 'count': words_counts}).sort_values(by=['count'], ascending=False)
        #print(df_words)
        keywords.append(df_words['word'].tolist()[0])
        keywordscounts.append(df_words['count'].tolist()[0])
        clusters.append(','.join(words))

    return keywords, pd.DataFrame.from_dict({'keyword': keywords, 'keyword counts': keywordscounts, 'clusters': clusters}).sort_values(by=['keyword counts'], ascending=False)

def get_entities(text):
    doc = nlp(text)
    entities,types = [],[]
    for ent in doc.ents:
        if ent.label_ in ENTITIES_TYPES:
           entities.append(ent.text)
           #types.append(ent.label_)
    return entities

def get_keyWords_cliques(cliques, counts):

    clusters, keywords, keywordscounts = [], [], []
    for i in range(len(cliques)):
        words = [Gm.vs[v]["name"] for v in cliques[i]]
        words_counts = [counts[w] for w in words]
        df_words = pd.DataFrame.from_dict({'word': words, 'count': words_counts}).sort_values(by=['count'], ascending=False)
        #print(df_words)
        keywords.append(df_words['word'].tolist()[0])
        keywordscounts.append(df_words['count'].tolist()[0])
        clusters.append(','.join(words))

    return keywords, pd.DataFrame.from_dict({'keyword': keywords, 'keyword counts': keywordscounts, 'clusters': clusters}).sort_values(by=['keyword counts'], ascending=False)



def get_final_kw_list(reviews):
    print('------ get reviews  nouns and noun phrases--------')
    nouns = get_nouns(reviews)
    noun_phrases = get_nounphrases(reviews)
    ####
    phrase_list = [phr.lower() for phr in nouns + noun_phrases]
    phrase_list = filter_phrase_list(phrase_list)
    print('------ Done filtering--------')
    phCounts = Counter(phrase_list)
    phrases = list(set(phrase_list))
    print('Done converting them to unique set')
    phrases_embeddings = sentenceEncoder.encode(phrases)
    print('Done encoding')
    simDF = get_similarityMatrix(phrases_embeddings, phrases)
    print('Done calculating similarity matrix')
    tuples = [tuple(x) for x in simDF.values]
    Gm = ig.Graph.TupleList(tuples, directed=False, edge_attrs=['sims'])
    print('Done generating graph')
    components = Gm.components()
    print('Done computing components')
    try:
       num_clusters = max(components.membership)
       keywords, df = get_keyWords(num_clusters, components, phCounts)
       df = df[df['keyword counts'] >= 1]

       kws = df['keyword'].tolist()
       final_kw_list = []
       for kw in kws:
           final_kw_list.append(' '.join([w for w in kw.split(' ') if w not in STOP_WORDS]))

       return final_kw_list

    except:
       return ''





reviews = '''
Came here to have my IPAD4 cracked screen fixed and they did a fantastic job and at a really affordable price. Definitely worth a try if you have a need to repair you tablet or phone. Will update if there are any problems but they have a 30 day warranty.
I stopped by Mobile Square recently, to replace the battery on my iPhone 6. The staff were professional, knowledgeable, and had the job done in 20 minutes! My phone has been working great. With the 30 day warranty they offer, plus their ridiculously good prices, I couldn't recommend this place enough.
Got my iPhone 6 wet on a run, power button didn't work, volume and home button on the fritz, and the phone kept trying to turn off. $30 and 1 hour later, I'm happily posting this from my phone!
These guys are the biggest crooks known to man. They fixed my screen with an aftermarket part claiming that was an original part. The resolution on my screen was distorted because of this.My touch screen is not glitching after 3 weeks.DO NOT GO TO THEM. THEY JUST WANT YOUR MONEY!! HORRIBLE SERVICE!
These guys were really quick and professional! Was recommended by a friend. Got my Samsung phone fixed in 30 mins and the price was great and service was attentive. They also threw in a case and protector. Now my phone feels great! Would totally recommend and definitely coming back!
Unreliable liar. Does not honour what their agreed upon quoted price was. I called them twice confirmed the price to replace a glass screen. After coming in, the price was changed on me. Beware. Do not deal with.
Awesome super fast super inexpensive service. I was really impressed. Really professional and nice people. I smashed my iPhone 7 screen to pieces and i

'''

'''
data = pd.read_csv('../FewSumm/artifacts/yelp/gold_summs/test.csv', delimiter='\t')
prompts = []
for ix in range(data.shape[0]):

    responses = [data['rev' + str(i)].tolist()[ix] for i in range(1, 9)]

    nouns, noun_phrases, entities = [], [], []
    for response in responses:
        # print(response)
        ents = get_entities(response)
        nouns.extend(get_nouns(response, ents))
        noun_phrases.extend(get_nounphrases(response, ents))
        entities.extend(ents)

    # print(len(entities),len(tps))
    # ner_df = pd.DataFrame.from_dict({'entities':entities,'entity_type':ent_types})
    phrase_list = [phr.lower() for phr in nouns + noun_phrases]
    phrase_list = filter_phrase_list(phrase_list)
    print('Done adding phrases')
    phCounts = Counter(phrase_list)
    phrases = list(set(phrase_list))

    # phrases = list(set(nouns))
    print('Done converting them to unique set')
    phrases_embeddings = sentenceEncoder.encode(phrases)
    print('Done encoding')
    simDF = get_similarityMatrix(phrases_embeddings, phrases)
    print('Done calculating similarity matrix')
    tuples = [tuple(x) for x in simDF.values]
    Gm = ig.Graph.TupleList(tuples, directed=False, edge_attrs=['sims'])
    print('Done generating graph')
    components = Gm.components()
    print('Done computing components')
    num_clusters = max(components.membership)
    keywords, df = get_keyWords(num_clusters, components, phCounts)
    df = df[df['keyword counts'] >= 3]
    prompts.append(' | '.join(df['keyword'].tolist())+' | reviews: ')


data['prompts'] = prompts

data.to_csv('../FewSumm/artifacts/yelp/gold_summs/test.csv', sep='\t', index=False)
'''
