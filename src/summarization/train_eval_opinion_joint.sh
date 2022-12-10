set -e

export CUDA_VISIBLE_DEVICES=1,2,3,4,5

k=200
dataset='yelp'
bankDir=~/public/datastore/yelp_bank

echo 'training started' $dataset '...'
python train_encoder_decoder.py --train $bankDir'/prompted_yelp_train.csv'  --valid $bankDir'/prompted_yelp_valid.csv' --config ./config/train_config.yml --outdir /afs/cs.pitt.edu/usr0/mse30/public/datastore/opinionsumm_models/$dataset'finetune_bart_prompted_ext_para_20k' --input_col review --summ_col summary

#echo 'testing started ..'
#python test_encoder_decoder.py --test ../../../FewSumm/artifacts/$dataset/gold_summs/sum_pairs/$dataset'_test_df.csv'  --prediction outputs/$dataset'_test_finetune_bart_large_extractive.csv' --model /afs/cs.pitt.edu/usr0/mse30/public/datastore/opinionsumm_models/yelpfinetune_bart_extractive/checkpoint-70 --config ./config/train_config.yml --input_col review

#echo 'transcrive unlabeled ...'
#for k in 100 200 300 400 500 600 700 800 900 1000; do
#   echo processing file ~/public/datastore/yelp_bank/nearest_neighbors_k$k'.csv'	
#python test_encoder_decoder.py --test ~/public/datastore/yelp_bank/nearest_neighbors_k$k'.csv'  --prediction ~/public/datastore/yelp_bank/nearest_neighbors_predictions_k$k'.csv' --model /afs/cs.pitt.edu/usr0/mse30/public/datastore/opinionsumm_models/$dataset'finetune_yelp_bart_large'/checkpoint-30  --config ./config/train_config.yml --input_col reviews
#done 

