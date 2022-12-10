import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from nltk.tokenize import sent_tokenize
import argparse
import swifter
from datasets import load_dataset, load_metric
import pandas as pd

my_parser = argparse.ArgumentParser()
my_parser.add_argument("-reviews", type=str)
my_parser.add_argument("-paraphrased", type=str)
args = my_parser.parse_args()

model_name = 'tuner007/pegasus_paraphrase'
## Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
## half is to convert the model to fp16
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda").half()



def generate_answer(batch):
  inputs_dict = tokenizer(batch['sentence'], padding="max_length",
                          max_length=50, return_tensors="pt", truncation=True)
  input_ids = inputs_dict.input_ids.to("cuda")
  attention_mask = inputs_dict.attention_mask.to("cuda")
  global_attention_mask = torch.zeros_like(attention_mask)
  # put global attention on <s> token
  global_attention_mask[:, 0] = 1

  predicted_abstract_ids = model.generate(
    input_ids, attention_mask=attention_mask)
  batch["paraphrased_sentence"] = tokenizer.batch_decode(
    predicted_abstract_ids, skip_special_tokens=True)
  return batch

#df = pd.DataFrame(data={'review':open(args.reviews,'r').readlines()})
df = pd.read_csv(args.reviews)
sentences =[]
rev_ids = []
for ix, rev in enumerate(df['reviews'].tolist()):
    sentences.extend(sent_tokenize(rev))
    rev_ids.extend(['reviews_id'+str(ix+1)]*len(sent_tokenize(rev)))

pd.DataFrame(data={'reviews_id':rev_ids,'sentence':sentences}).to_csv('~/public/datastore/yelp_bank/unannotated_review_sentences_temp_chunk.csv', index=False)
test_doc = load_dataset(
    "csv", data_files='/afs/cs.pitt.edu/usr0/mse30/public/datastore/yelp_bank/unannotated_review_sentences_temp_chunk.csv')['train']

result = test_doc.map(generate_answer, batched=True, batch_size=16)

res_df = pd.DataFrame(data={'review_id':result['review_id'], 'sentence':result['sentence'], 'paraphrased_sentence': result['paraphrased_sentence']})
original,paraphrased = [],[]

for  review_id, review_df in res_df.groupby(by=['review_id']):
     original.append(' '.join(review_df['sentence'].tolist()))
     paraphrased.append(' '.join(review_df['paraphrased_sentence'].tolist()))

pd.DataFrame(data={'reviews':original, 'paraphrased_review':paraphrased}).to_csv(args.paraphrased, index=False)
