set -e

export CUDA_VISIBLE_DEVICES=6,7

dataset='amazon'

echo 'training started' $dataset '...'
python train_encoder_decoder.py --train ../../../FewSumm/artifacts/$dataset/gold_summs/sum_pairs/$dataset'_train_prompted_df.csv'  --valid ../../../FewSumm/artifacts/$dataset/gold_summs/sum_pairs/$dataset'_valid_prompted_df.csv' --config ./config/train_config.yml --outdir /afs/cs.pitt.edu/usr0/mse30/public/datastore/opinionsumm_models/$dataset'finetune_yelp_bartprompted_chkpt3k_promptedsumm' --input_col prompted_reviews --summ_col summary

#echo 'testing started ..'
#python test_encoder_decoder.py --test ../../../FewSumm/artifacts/$dataset/gold_summs/sum_pairs/$dataset'_test_prompted_df.csv'  --prediction outputs/$dataset'_test_finetune_promptedbart_chkpt3k.csv' --model /afs/cs.pitt.edu/usr0/mse30/public/datastore/opinionsumm_models/$dataset'finetune_yelp_bartprompted_chkpt3k_promptedsumm'/checkpoint-30 --config ./config/train_config.yml --input_col prompted_reviews

#echo 'transcrive unlabeled ...'
#for k in 100 200 300 400 500 600 700 800 900 1000; do
#   echo processing file ~/public/datastore/yelp_bank/nearest_neighbors_k$k'.csv'	
#python test_encoder_decoder.py --test ~/public/datastore/yelp_bank/nearest_neighbors_k$k'.csv'  --prediction ~/public/datastore/yelp_bank/nearest_neighbors_predictions_k$k'.csv' --model /afs/cs.pitt.edu/usr0/mse30/public/datastore/opinionsumm_models/$dataset'finetune_yelp_bart_large'/checkpoint-30  --config ./config/train_config.yml --input_col reviews
#done 

