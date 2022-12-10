import pandas as pd
from ast import literal_eval

course='ie256_2016'

df_test_with_concepts = pd.read_csv(
    ''.join(['../courseMirror/courseMirrorSummarization/CourseMirror_data/sum_pairs/',course,'_with_concept_reflections.csv'])).rename(
    columns={'summary': 'oracle'})
df_beam1 = pd.read_csv(''.join(['../courseMirror/src/summarization/outputs/',course ,'_bart_clarge_retested_beam1.csv'])).rename(
    columns={'generated_summary': 'generated_summary_beam1'})
df_beam2 = pd.read_csv(''.join(['../courseMirror/src/summarization/outputs/', course,'_bart_clarge_retested_beam2.csv'])).rename(
    columns={'generated_summary': 'generated_summary_beam2'})
df_beam3 = pd.read_csv(''.join(['../courseMirror/src/summarization/outputs/', course,'_bart_clarge_retested_beam3.csv'])).rename(
    columns={'generated_summary': 'generated_summary_beam3'})
df_beam4 = pd.read_csv(''.join(['../courseMirror/src/summarization/outputs/', course,'_bart_clarge_retested_beam4.csv'])).rename(
    columns={'generated_summary': 'generated_summary_beam4'})
df_beam5 = pd.read_csv(''.join(['../courseMirror/src/summarization/outputs/', course,'_bart_clarge_retested_beam5.csv'])).rename(
    columns={'generated_summary': 'generated_summary_beam5'})

df_test = pd.merge(df_test_with_concepts, df_beam1, on='oracle')
df_test = pd.merge(df_test, df_beam2, on='oracle')
df_test = pd.merge(df_test, df_beam3, on='oracle')
df_test = pd.merge(df_test, df_beam4, on='oracle')
df_test = pd.merge(df_test, df_beam5, on='oracle')

df_test.to_csv('../courseMirror/src/summarization/outputs/'+course+'_bart_large_multiple_beams_concepts.csv', index=False)
