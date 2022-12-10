set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8

echo 'training started ...'
#python train_encoder_decoder.py --train ../../courseMirrorSummarization/CourseMirror_data/sum_pairs/train_except_ie256.csv --valid ../../courseMirrorSummarization/CourseMirror_data/sum_pairs/valid_except_ie256.csv --config ./config/train_config.yml --outdir /afs/cs.pitt.edu/usr0/mse30/public/datastore/opinionsumm_models/coursemirror_models/ie256_bart_cnn_model --input_col reference --summ_col summary


echo 'test started ...'
python test_encoder_decoder.py --test ../../courseMirrorSummarization/CourseMirror_data/sum_pairs/ie256.csv --prediction outputs/ie256_bart_cnn_summaries.csv --model /afs/cs.pitt.edu/usr0/mse30/public/datastore/opinionsumm_models/coursemirror_models/ie256_bart_cnn_model/checkpoint-60 --config ./config/train_config.yml --input_col reference
