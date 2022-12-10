import os
import pandas as pd
import data_helper as dh
from sklearn.model_selection import train_test_split

data_dir = '../../courseMirrorSummarization/CourseMirror_data/stories_salem'
out_base_dir = '../../courseMirrorSummarization/CourseMirror_data'

cs0445_path = os.path.join(data_dir, 'CS0445')
engr_path = os.path.join(data_dir, 'ENGR')
ie256_path = os.path.join(data_dir, 'IE256')
ie256_2016_path = os.path.join(data_dir, 'IE256_2016')

def construct_ref_summ_pair(data_path):
    ref, summ = [], []
    for file in os.listdir(data_path):
        ref.append(dh.get_reflection(os.path.join(data_path, file)))
        summ.append(dh.get_abs_summ(os.path.join(data_path, file)))
    return pd.DataFrame(data={'reflection': ref, 'summary': summ})

data_cs = construct_ref_summ_pair(data_path=cs0445_path)
data_engr = construct_ref_summ_pair(data_path=engr_path)
data_ie256 = construct_ref_summ_pair(data_path=ie256_path)
data_ie256_2016 = construct_ref_summ_pair(data_path=ie256_2016_path)

df_full_courses_data = pd.concat([data_cs, data_engr, data_ie256, data_ie256_2016])
#####

df_train, df_rest = train_test_split(df_full_courses_data, test_size=0.2)
df_valid, df_test = train_test_split(df_rest, test_size=0.5)

##### dump csv ####
df_train.to_csv('../../courseMirrorSummarization/CourseMirror_data/sum_pairs/courses_full_train.csv', index=False)
df_valid.to_csv('../../courseMirrorSummarization/CourseMirror_data/sum_pairs/courses_full_valid.csv', index=False)
df_test.to_csv('../../courseMirrorSummarization/CourseMirror_data/sum_pairs/courses_full_test.csv', index=False)



