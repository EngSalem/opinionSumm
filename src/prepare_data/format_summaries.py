import os
import pandas as pd
import data_helper as dh

data_dir = '../../courseMirrorSummarization/CourseMirror_data/stories_salem'
out_base_dir = '../../courseMirrorSummarization/CourseMirror_data'

## old data used at EMBLP submission
data_cs0445 = os.path.join(data_dir, 'CS0445')
data_engr = os.path.join(data_dir, 'ENGR')
data_ie256 = os.path.join(data_dir, 'IE256')
data_ie256_2016 = os.path.join(data_dir, 'IE256_2016')


def construct_ref_summ_pair(data_path):
    ref, summ = [], []
    for file in os.listdir(data_path):
        ref.append(dh.get_reflection(os.path.join(data_path, file)))
        summ.append(dh.get_abs_summ(os.path.join(data_path, file)))
    return pd.DataFrame(data={'reflection': ref, 'summary': summ})

def construct_prompt(summary, concepts):
    included_concepts = []
    for concept in concepts:
        if concept in summary:
           included_concepts.append(concept)

    return ' | '.join(included_concepts) + ' | '

def get_existing_concepts(review, concepts):
    return [concept for concept in concepts if concept in review]



def construct_ref_summ_pair(data_path):
    ref,summ  = [], []

    for file in os.listdir(data_path):
        reflections = dh.get_reflection(os.path.join(data_path, file))
        summary = dh.get_abs_summ(os.path.join(data_path, file))
        concepts = dh.get_concept(os.path.join(data_path, file))
        prompt = construct_prompt(summary,concepts)
        ref.append( ' '.join([prompt,'reflections:',reflections]))
        summ.append(summary)

    return pd.DataFrame.from_dict({'prompted_reflections': ref,
                                  'summary': summ})


def construct_ref_summ_concepts_triplets(data_path):
    ref, summ, final_concepts = [], [], []

    for file in os.listdir(data_path):
        reflections = dh.get_reflection(os.path.join(data_path, file))
        summary = dh.get_abs_summ(os.path.join(data_path, file))
        concepts = dh.get_concept(os.path.join(data_path, file))
        ##
        review_concepts = get_existing_concepts(reflections, concepts)

        ref.append(reflections)
        summ.append(summary)
        final_concepts.append(review_concepts)
    print(len(concepts), len(ref), len(summ))


    return pd.DataFrame(data={'reflections': ref, 'summary':summ, 'concepts': final_concepts})






df_summ_ref_cs0445 = construct_ref_summ_concepts_triplets(data_path=data_cs0445)
df_summ_ref_ie256 = construct_ref_summ_concepts_triplets(data_path=data_ie256)
df_summ_ref_ie256_2016 = construct_ref_summ_concepts_triplets(data_path=data_ie256_2016)
df_summ_ref_engr = construct_ref_summ_concepts_triplets(data_path=data_engr)

df_summ_ref_cs0445.to_csv(os.path.join('../../courseMirrorSummarization/CourseMirror_data/sum_pairs/cs0445_with_concept_reflections.csv'), index=False)
df_summ_ref_ie256.to_csv(os.path.join('../../courseMirrorSummarization/CourseMirror_data/sum_pairs/ie256_with_concept_reflections.csv'), index=False)
df_summ_ref_ie256_2016.to_csv(os.path.join('../../courseMirrorSummarization/CourseMirror_data/sum_pairs/ie256_2016_with_concept_reflections.csv'), index=False)
df_summ_ref_engr.to_csv(os.path.join('../../courseMirrorSummarization/CourseMirror_data/sum_pairs/engr_with_concept_reflections.csv'), index=False)
