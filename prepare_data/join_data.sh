set -e

bankDir='~/public/datastore/yelp_bank'
trainDir='~/public/Summarization/OpinionSummarization/FewSumm/artifacts/yelp/gold_summs/sum_pairs/'
dataset='yelp'

for k in 100 200 300 400; do
  python join_predicted_manual.py --predicted $bankDir'/nearest_neighbors_predictions_k'$k'.csv' \
  --nn_file $bankDir'/nearest_neighbors_k'$k'.csv' --train $trainDir'/'$dataset'_train_df.csv' --combined $bankDir'/'$dataset'_train_combined_k'$k'.csv'

done

