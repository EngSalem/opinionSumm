import pandas as pd

df = pd.read_csv('../courseMirror/src/summarization/yelp_test_df_pertubed_inputs_outs.csv')

df_original, df_k1, df_k2, df_k3 = [], [], [], []

for review_id, df_rev_id in df.groupby(by=['group_id']):

    if '_pertubed_1' in review_id:
        df_k1.append(df_rev_id)

    elif '_pertubed_2' in review_id:
        df_k2.append(df_rev_id)

    elif '_pertubed_3' in review_id:
        df_k3.append(df_rev_id)

    else:
        df_original.append(df_rev_id)



df_original = pd.concat(df_original)
print(df_original.shape)
df_k1 = pd.concat(df_k1)
print(df_k1.shape)
df_k2 = pd.concat(df_k2)
print(df_k2.shape)
df_k3 = pd.concat(df_k3)
print(df_k3.shape)
## merge all into ond dataframe
df_k1['temp_id'] = df_k1['group_id'].apply(lambda id: id.split('_pertubed_')[0])
df_k2['temp_id'] = df_k2['group_id'].apply(lambda id: id.split('_pertubed_')[0])
df_k3['temp_id'] = df_k3['group_id'].apply(lambda id: id.split('_pertubed_')[0])

df_original = df_original.merge(df_k1[['temp_id', 'generated_summary']].rename(columns={'generated_summary': 'generated_summary_pertubed_1'}),
                  left_on='group_id', right_on='temp_id').drop(columns=['temp_id'])


df_original = df_original.merge(df_k2[['temp_id', 'generated_summary']].rename(columns={'generated_summary': 'generated_summary_pertubed_2'}),
                  left_on='group_id', right_on='temp_id').drop(columns=['temp_id'])

df_original = df_original.merge(df_k3[['temp_id', 'generated_summary']].rename(columns={'generated_summary': 'generated_summary_pertubed_3'}),
                  left_on='group_id', right_on='temp_id').drop(columns=['temp_id'])

df_original.drop_duplicates().to_csv('../courseMirror/src/summarization/outputs/yelp_test_df_pertubed_inputs_outs_formatted.csv', index=False)