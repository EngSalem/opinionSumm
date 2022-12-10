from transformers import MarianMTModel, MarianTokenizer
import argparse
import swifter

parser = argparse.ArgumentParser()
parser.add_argument('--input', action='store', dest='input',
                    help='')
parser.add_argument('--nn_file', action='store', dest='nn_file',
                    help='')
parser.add_argument('--output', action='store', dest='output', help='output reviews')
args = parser.parse_args()

target_model_name = 'Helsinki-NLP/opus-mt-en-ROMANCE'
en_model_name = 'Helsinki-NLP/opus-mt-ROMANCE-en'
target_tokenizer = MarianTokenizer.from_pretrained(target_model_name)
target_model = MarianMTModel.from_pretrained(target_model_name)
en_tokenizer = MarianTokenizer.from_pretrained(en_model_name)
en_model = MarianMTModel.from_pretrained(en_model_name)


def translate(texts, model, tokenizer, language="fr"):
    # Prepare the text data into appropriate format for the model
    template = lambda text: f"{text}" if language == "en" else f">>{language}<< {text}"
    src_texts = [template(text) for text in texts]

    # Tokenize the texts
    encoded = tokenizer.prepare_seq2seq_batch(src_texts)

    # Generate translation using model
    translated = model.generate(**encoded)

    # Convert the generated tokens indices back into text
    translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)

    return translated_texts


def back_translate(texts, source_lang="en", target_lang="fr"):
    # Translate from source to target language
    fr_texts = translate(texts, target_model, target_tokenizer,
                         language=target_lang)

    # Translate from target language back to source language
    back_translated_texts = translate(fr_texts, en_model, en_tokenizer,
                                      language=source_lang)

    return back_translated_texts

## read predicted
df_predicted = pd.read_csv(args.input)
## back translate
df_predicted['backtranslated'] = back_translate(texts=df_predicted['generated_summary'].tolist())
## read nn
df_nn = pd.read_csv(args.nn_file)

pd.concat([pd.DataFrame({'review':df_nn['reviews'].tolist(),'summary':df_predicted['generated_summary'].tolist()}),
pd.DataFrame({'review':df_nn['reviews'].tolist(),'summary':df_predicted['backtranslated'].tolist()})]).to_csv(args.output)


