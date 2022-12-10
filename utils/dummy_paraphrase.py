import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import argparse
from nltk.tokenize import sent_tokenize

my_parser = argparse.ArgumentParser()
my_parser.add_argument("-reviews", type=str)
my_parser.add_argument("-paraphrased", type=str)
args = my_parser.parse_args()

model_name = 'tuner007/pegasus_paraphrase'
## Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
## half is to convert the model to fp16
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda").half()

def generate_paraphrase(sentence):
  batch = tokenizer(sentence, return_tensors='pt').to("cuda")
  generated_ids = model.generate(batch['input_ids'])
  generated_sentence = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
  return generated_sentence

def get_paraphrased_review(review):
    sents = sent_tokenize(review)
    paraphrased_sents = []
    for sent in sents:
        paraphrased_sents.extend(generate_paraphrase(sent))
    
    print(paraphrased_sents)
    return '\n'.join(paraphrased_sents)

review = '''
LOVE! This freakin place has an amazing decor, friendly staff..and the food was really yummy! I went during restaurant week, so the deal was great! A lot of food for the $ and the soups were great. Whoa! Pretentious aren't we. Wed night 6pm, drop in for early dinner. Two people at the bar and two tables of two diners.  Estimate 20-25 empty tables. "...sorry sir, we can't seat you until 7:15..." Good luck, hope your delusions of grandeur are one day realized. Went here with a large group (12) for brunch one day and had an excellent experience. A mix of vegetarians, vegans, picky eaters and people who didnt like Indian food and no one walked away unhappy. Service was excellent, as was the food. I came here with my family during Christmas weekend. We sat down for brunch and absolutely loved it. Laurie was our waitress and she was great, very helpful and accommodating. We did endless mimosas and she was always on point about refilling our glasses.   Food- Paneer Burji, Spinach Chaat, Paneer Pizza are a must. This was probably one of the best burjis I have head, it comes with a chili naan and you can customize it to your spice level. Great for happy hour, so-so for brunch.   5/5 for Happy hour:  They have wonderful drink specials and great deals on small plates. The chikni shandar (chai sangria) is my go to drink here, and the naan pizza never disappoints. I'm also a huge fan of their stuffed long hot peppers, but they tend to run a little spicy for people who aren't used to burning their taste buds on a daily basis. Also, Indeblue is never too crowded during happy hour, and I've always been able to grab a seat even if I'm with a group of friends.  3/5 for Brunch: The paneer bhurjee is my favorite brunch item here, not only because it's delicious and cheesy, but also because it's one of the few brunch items at indeblue that are actually filling. The "build your own" omelette was the smallest omelette I've ever seen, as well as the French toast portion. Everything here tastes great, it's just that some portions are a bit lacking. Also, their brunch starts a little late -- 11am, so if you're and early riser looking to beat the brunch crowd, this is not the place to be.  Overall-- indeblue is one of my favorite happy hour places in center city, but I'm not too excited about their brunch. I am always on the hunt for my third favorite cuisine--Indian!  Now, I am usually inclined to dine at a more traditional style indian Restaraunt, but IndeBlue does Indian-fusion very well.  Service was quite attentive (thanks to Josh), which is the expectation with Indian restaurants.  After being sat, we were each served a small portion mulligatawny soup-delicious, but it was not on our list if must-try's for our first visit.   Please start with Crispy Spinach Chaat--it will not disappoint!  Large enough portion for four to share.  We also ordered Lasooni Naan with the scallops. There were only three scallops, but the sauce is a wonderful addition (but could be improved by reducing slightly longer to thicken).  We ordered Laal Maas,  Malwani Swordfish, Seafood Moilee, and Chicken Madras. The Seafood Moilee was the standout, but everything was wonderful; EXCEPT the sword fish: it was overcooked (dry), but the Makwani sauce was a great addition.  Although, I did not partake (way too full), the table greatly enjoyed Banana Nirvana.  We will return an hopefully soon! Brunch was not a meal but an experience! The sitarist played the most beautiful music transporting the meal all the way to India! The chai tea was like nectar of the gods and the non vegetarian thali was delightful! Will definitely come back next time I'm in Philadelphia! One of my favorite happy hours in the city (5-7 and then after 9, daily!). This place gets crowded around the bar, but it's not hard to snatch a seat or two if you have a small party. Intense physical proximity during peak hours doesn't make for the best heart-to-heart's (unless you opt for the quieter dining area), but it's great for delicious bites and strong drinks after work.   Recommended noms: Drums of heaven - these are so good! A bit spicier than expected but like firecrackers in your mouth. Great HH snack though beware as it gets messy.  Lamb rogan josh (19) - delicious  lamb dish, rich in spices and flavor. Served with rice but great with their garlic naan. Split this with friend and we were very satisfied.   Drinks: Kingfisher lager - light and refreshing, complemented the meal well.  Indebluetini - rich mango flavor! favorite.  Rose gimlet - pretty and strong, but a bit lackluster in taste.   Take advantage of their long happy hours to sample their menu and you won't be disappointed!'
'''
print(get_paraphrased_review(review))


