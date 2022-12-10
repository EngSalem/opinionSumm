set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4

dataset=yelp
k=300

echo ~/public/datastore/$dataset'_bank/nearest_neighbors_predictions_k'$k'_.csv'
python train_encoder_decoder.py --train ~/public/datastore/$dataset'_bank/nearest_neighbors_predictions_k'$k'.csv'  --valid ../../../FewSumm/artifacts/$dataset/gold_summs/sum_pairs/$dataset'_valid_df.csv' --config ./config/train_config.yml --outdir /afs/cs.pitt.edu/usr0/mse30/public/datastore/opinionsumm_models/$dataset'_models_bart_finetune_psudolabels_k'$k --input_col review --summ_col summary


