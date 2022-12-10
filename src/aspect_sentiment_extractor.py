import pandas as pd
import spacy
# import aspect_extractor as ae
# import swifter
import stanza
import argparse
from transformers import pipeline
from opinionSumm.src.parse_xml import *
from sklearn.metrics import f1_score
import inflect

# initialize spacy en model
inflect = inflect.engine()

classifier = pipeline("zero-shot-classification",
							  model="facebook/bart-large-mnli")
candidate_labels = ['positive', 'negative', 'neutral']


stanza.download('en')
nlp = stanza.Pipeline('en')


# rules from paper: https://arxiv.org/abs/2109.03821
def get_aspect_sentiment_candidates_original(reviews):
	print(type(reviews))
	doc = nlp(reviews)
	## iterate over sentences
	candidates = []
	for sent in doc.sentences:
		## get dependencies
		for i in range(len(sent.dependencies)):
			if (sent.dependencies[i][-1].upos == 'ADJ' and sent.dependencies[i][1] == 'amod' and sent.dependencies[i][
				0].upos == 'NOUN'):
				candidates.append((sent.dependencies[i][-1].text, sent.dependencies[i][0].text))

			elif (sent.dependencies[i][-1].upos == 'NOUN' and sent.dependencies[i][1] == 'nsubj' and
				  sent.dependencies[i][0].upos == 'ADJ'):
				candidates.append((sent.dependencies[i][0].text, sent.dependencies[i][-1].text))

	return candidates

## Rules for aspect sentiment pairs: R1:  (NN) -> nsubj -> (adj) | R2: (ADJ) -> amod (NOUN)
## Rules for merging aspect words: R3: (NN) ->nn -> (NN) | R4:
## Rules for merging sentiment words: R5: (ADJ)-> conj (ADJ)
def get_dependencies(text):
	doc = nlp(text)
	
	final_candidates = []

	## iterate over sentences
	for sent in doc.sentences:
		candidates = []
		sentiment_word_pairs = []
		aspect_word_pairs = []

		for i, dependency in enumerate(sent.dependencies):
			############ rules for aspect-sentiment pairs
			## apply rule 1
			if ((sent.dependencies[i][-1].upos == 'NOUN' or sent.dependencies[i][-1].upos == 'PROPN') and
					sent.dependencies[i][0].upos == 'ADJ' and \
					sent.dependencies[i][1] == 'nsubj'):
				candidates.append((sent.dependencies[i][0].text, sent.dependencies[i][-1].text))
			## apply rule 2
			elif sent.dependencies[i][-1].upos == 'ADJ' and (
					sent.dependencies[i][0].upos == 'NOUN' or sent.dependencies[i][0].upos == 'PROPN') and \
					sent.dependencies[i][1] == 'amod':
				candidates.append((sent.dependencies[i][-1].text, sent.dependencies[i][0].text))

			########### rules for aspect-words #########
			## appply rule 3

			elif (sent.dependencies[i][-1].upos == 'PROPN' or sent.dependencies[i][-1].upos == 'NOUN') and (
					sent.dependencies[i][0].upos == 'PROPN' or sent.dependencies[i][-1].upos == 'NOUN') and \
					sent.dependencies[i][1] == 'compound':
				aspect_word_pairs.append((sent.dependencies[i][-1].text, sent.dependencies[i][0].text))

			## apply rule 4
			elif (sent.dependencies[i][-1].upos == 'PROPN' or sent.dependencies[i][-1].upos == 'NOUN') and (
					sent.dependencies[i][0].upos == 'PROPN' or sent.dependencies[i][-1].upos == 'NOUN') and \
					sent.dependencies[i][1] == 'conj':
				aspect_word_pairs.append(
					(sent.dependencies[i][-1].text, sent.dependencies[i][0].text))

			####### rules for multiple sentiment words
			## apply rule 5

			elif sent.dependencies[i][-1].upos == 'ADJ' and sent.dependencies[i][0].upos == 'ADJ' and \
					sent.dependencies[i][1] == 'conj':
				sentiment_word_pairs.append(
					(sent.dependencies[i][-1].text, sent.dependencies[i][0].text))

		#####  merge the aspect words and expand

		for candidate in candidates:
			added = False

			curr_candidates = []
			for aspect_pair in aspect_word_pairs:
				## adjust aspect pairs
				if candidate[1] in aspect_pair:
					added = True
					final_candidates.append((candidate[0], aspect_pair[0] + ' ' + aspect_pair[1]))
					curr_candidates.append((candidate[0], aspect_pair[0] + ' ' + aspect_pair[1]))

			if not added:
				final_candidates.append(candidate)
				curr_candidates.append(candidate)

			for sentiment_pair in sentiment_word_pairs:
				## expand sentiment list
				for curr_candidate in curr_candidates:
					if curr_candidate[0] in sentiment_pair:
						final_candidates.extend([(sentiment_pair[0], curr_candidate[-1]), (sentiment_pair[1], curr_candidate[-1])])
	print('final candidates: ', final_candidates)
	return set(final_candidates)

# rules from paper with added merging of adjectives to the aspect
# merges all adjectives to the aspect except for the first one in the case of R1: (NN) -> nsubj -> (adj)
# Ex. BEST spicy tuna roll, great asian salad. -> (BEST, spicy tuna roll), (great, asian salad)
def get_aspect_sentiment_candidates_1(reviews):
	print(type(reviews))
	doc = nlp(reviews)
	## iterate over sentences
	final_candidates = []
	for sent in doc.sentences:
		candidates = []
		candidates_ids = []
		merge_compounds = []
		## get dependencies
		for i in range(len(sent.dependencies)):
	
			if (sent.dependencies[i][-1].upos == 'ADJ' and sent.dependencies[i][1] == 'amod' and sent.dependencies[i][
				0].upos == 'NOUN'):
				# check if noun is already in candidates_ids
				is_in = False
				for ci in range(len(candidates_ids)):
					if candidates_ids[ci][1][-1] == sent.dependencies[i][0].id:
						is_in = True
						candidates_ids[ci][1].insert(-1, sent.dependencies[i][-1].id)

				if not is_in:
					candidates_ids.append([[sent.dependencies[i][-1].id], [sent.dependencies[i][0].id]])

			elif sent.dependencies[i][1] == 'compound':
				merge_compounds.append((sent.dependencies[i][-1].id, sent.dependencies[i][0].id))

			elif (sent.dependencies[i][-1].upos == 'NOUN' and sent.dependencies[i][1] == 'nsubj' and
				  sent.dependencies[i][0].upos == 'ADJ'):
				candidates_ids.append([[sent.dependencies[i][0].id], [sent.dependencies[i][-1].id]])

		# put compounds into candidates_ids
		merge_compounds_main_noun = [merge_comp[1] for merge_comp in merge_compounds]
		for i in range(len(candidates_ids)):
			if candidates_ids[i][1][-1] in merge_compounds_main_noun: # only main nouns can have compounds so we only search the last word of the aspect
				index = merge_compounds_main_noun.index(candidates_ids[i][1][-1])
				candidates_ids[i][1].insert(-1, merge_compounds[index][0])
		# convert candidates_ids into candidates
		candidates_ids_new = candidates_ids.copy()

		for i in range(len(candidates_ids)):
			for j in range(len(candidates_ids[i])):
				for k in range(len(candidates_ids[i][j])):
					candidates_ids_new[i][j][k] = sent.words[int(candidates_ids[i][j][k])-1].text # -1 because word ids are indexed starting from 1

		for i in range(len(candidates_ids_new)):
			sentiment = ' '.join(candidates_ids_new[i][0])
			aspect = ' '.join(candidates_ids_new[i][1])
			candidates.append((sentiment, aspect))
		final_candidates = final_candidates + candidates
	# print(final_candidates)
	return final_candidates

# rules from get_aspect_sentiment_candidates_1
# merges all adjectives to the aspect in the case of R2: (ADJ) -> amod (NOUN)
# Ex. The Japanese rice was amazing -> (Japanese rice, amazing)
# before, get_aspect_sentiment_candidates_1 would give us (Japanese, amazing), (rice, amazing)
def get_aspect_sentiment_candidates_2(reviews):
	print(type(reviews))
	doc = nlp(reviews)
	## iterate over sentences
	final_candidates = []
	for sent in doc.sentences:
		candidates = []
		candidates_ids = []
		rules = [] # corresponding array of each rule used in candidates_id 
		merge_compounds = []
		## get dependencies
		for i in range(len(sent.dependencies)):
	
			if (sent.dependencies[i][-1].upos == 'ADJ' and sent.dependencies[i][1] == 'amod' and sent.dependencies[i][
				0].upos == 'NOUN'):
				# check if noun is already in candidates_ids
				is_in = False
				for ci in range(len(candidates_ids)):
					if candidates_ids[ci][1][-1] == sent.dependencies[i][0].id:
						is_in = True
						candidates_ids[ci][1].insert(-1, sent.dependencies[i][-1].id)

				if not is_in:
					candidates_ids.append([[sent.dependencies[i][-1].id], [sent.dependencies[i][0].id]])
					rules.append(1) # amod is 1st rule

			elif sent.dependencies[i][1] == 'compound':
				merge_compounds.append((sent.dependencies[i][-1].id, sent.dependencies[i][0].id))

			elif (sent.dependencies[i][-1].upos == 'NOUN' and sent.dependencies[i][1] == 'nsubj' and
				  sent.dependencies[i][0].upos == 'ADJ'):
				candidates_ids.append([[sent.dependencies[i][0].id], [sent.dependencies[i][-1].id]])
				rules.append(2)

		# merge if we have nsubj rule and amod rule on the same noun Ex. The Japanese food was delicious-->(Japanese, food), (delicious, food)
		i = 0
		while i < len(candidates_ids):
			# check the last elem of aspect for main noun
			target_main_noun = candidates_ids[i][1][-1]
			target_rule = rules[i]
			ci = i
			while ci < len(candidates_ids):
				main_noun = candidates_ids[ci][1][-1]
				rule = rules[ci]
				if target_main_noun == main_noun and target_rule != rule: # condition for merging
					target_aspect = candidates_ids[i][1]
					print('target aspect: ', target_aspect)
					aspect = candidates_ids[ci][1]
					print('aspect: ', aspect)
					if target_rule == 1:
						new_aspect = candidates_ids[i][0] + target_aspect
						new_sentiment = candidates_ids[ci][0]
						candidates_ids[i] = [new_sentiment, new_aspect]
					else:
						new_aspect = candidates_ids[ci][0] + aspect
						new_sentiment = candidates_ids[i][0]
						candidates_ids[i] = [new_sentiment, new_aspect]
					candidates_ids.pop(ci)
				else:
					ci += 1
			i += 1

		# put compounds into candidates_ids
		merge_compounds_main_noun = [merge_comp[1] for merge_comp in merge_compounds]
		for i in range(len(candidates_ids)):
			if candidates_ids[i][1][-1] in merge_compounds_main_noun: # only main nouns can have compounds so we only search the last word of the aspect
				index = merge_compounds_main_noun.index(candidates_ids[i][1][-1])
				candidates_ids[i][1].insert(-1, merge_compounds[index][0])
		# convert candidates_ids into candidates
		candidates_ids_new = candidates_ids.copy()

		for i in range(len(candidates_ids)):
			for j in range(len(candidates_ids[i])):
				for k in range(len(candidates_ids[i][j])):
					candidates_ids_new[i][j][k] = sent.words[int(candidates_ids[i][j][k])-1].text # -1 because word ids are indexed starting from 1

		for i in range(len(candidates_ids_new)):
			sentiment = ' '.join(candidates_ids_new[i][0])
			aspect = ' '.join(candidates_ids_new[i][1])
			candidates.append((sentiment, aspect))
		final_candidates = final_candidates + candidates
	print(final_candidates)
	return final_candidates

def convert_sentiment_words_to_labels(candidates):
	final_candidates = []
	for candidate in candidates:
		label = filter_aspect_sentiment(candidate)
		if label is not None:
			final_candidates.append((candidate[1], label))
	return final_candidates

def filter_candidates(reviews, dep_list):
	## get keyword list
	kw_list = ae.get_final_kw_list(reviews)
	final_pairs = []
	for dep in dep_list:
		if dep[1] in kw_list and filter_aspect_sentiment((' '.join([dep[0], dep[1]]))):
			final_pairs.append(','.join([dep[0], dep[1]]))
	return final_pairs


# def get_aspect_Sentiment_prompts(reviews):
# 	dep_candidates = get_aspect_sentiment_candidates(reviews)
# 	if len(dep_candidates) > 0:
# 		final_dep_list = filter_candidates(reviews, dep_candidates)
# 	return ' | '.join(final_dep_list)


def get_aspect_sentiment_pairs_only(reviews, summary=False):
	dep_candidates = get_aspect_sentiment_candidates_1(reviews)

	if len(dep_candidates) > 0 and not summary:
		# final_dep_list = filter_candidates(reviews, dep_candidates)
		# return final_dep_list
		final_candidates = []
		for candidate in dep_candidates:
			label = filter_aspect_sentiment(candidate)
			if label is not None:
				final_candidates.append((candidate[1], label))
		return final_candidates

	elif len(dep_candidates) > 0 and summary:
		final_dep_list = [','.join([dep[0], dep[1]]) for dep in dep_candidates if filter_aspect_sentiment(dep)]
		return final_dep_list
	return dep_candidates


# Count number of overlapping aspect sentiment pairs
# Aspects that have different plurality are considered the same
def aspect_sentiment_differences(as1, as2):
	num_overlap = 0
	i = 0
	while i < len(as1):
		singular_as1 = (inflect.singular_noun(as1[i][0]), as1[i][1])
		plural_as1 = (inflect.plural_noun(as1[i][0]), as1[i][1])

		if as1[i] in as2 or singular_as1 in as2 or plural_as1 in as2: # overlap
			if as1[i] in as2:
				as2_i = as2.index(as1[i])
				print('OVERLAP!!!')
			elif singular_as1 in as2:
				as2_i = as2.index(singular_as1)
				print('OVERLAP, different plurality: ', singular_as1)
			else:
				as2_i = as2.index(plural_as1)
				print('OVERLAP, different plurality: ', plural_as1)
			as2.pop(as2_i)
			as1.pop(i)
			num_overlap += 1
		else:
			i+=1
	return num_overlap, len(as1), len(as2)

def filter_aspect_sentiment(aspect_sentiment):
	labels = classifier(aspect_sentiment, candidate_labels, device=0)
	# if (labels['labels'][0] == 'positive' or labels['labels'][0] == 'negative') and labels['scores'][0] >= 0.8:

	return labels['labels'][0]
	# if labels['scores'][0] >= 0.8:
	#     return labels['labels'][0]
	# else:
	#     return None

###################################
# GENERATE AS COLUMN IN DATA
###################################

def get_as_from_data(file, out_file):
	df_data = pd.read_csv(file)
	df_data['aspect_sentiment_for_review'] = df_data.apply(lambda row: get_aspect_sentiment_pairs_only(row['review'], True), axis=1)
	df_data['aspect_sentiment_for_summary'] = df_data.apply(lambda row: get_aspect_sentiment_pairs_only(row['summary'], True), axis=1)

	df_data.to_csv(out_file, index=False)


# generates output files containing aspects and sentiments 
def generate_data(data_string = None):
	file = './OpinionSummarization/FewSumm/artifacts/yelp/gold_summs/sum_pairs/yelp_train_df.csv'
	out_file = './generated_as_pairs_yelp_train.csv'
	if data_string:
		file = file.replace('train', data_string)
		out_file = out_file.replace('train', data_string)
	get_as_from_data(file, out_file)
	file = file.replace('yelp', 'amazon')
	out_file = out_file.replace('yelp', 'amazon')
	get_as_from_data(file, out_file)

###################################
# EVALUATION
###################################

# evaluate how well aspect sentiment extractor does by computing fscore between true AS pairs from dataset and generated AS pairs
def aspect_sentiment_extractor_eval():
	reviews, true_as_pairs = parse_xml()
	reviews = ' '.join(reviews) # concatenate reviews in one giant string
	candidates = get_aspect_sentiment_candidates_1(reviews)
	final_candidates = convert_sentiment_words_to_labels(candidates)
	print('GENERATED AS PAIRS: ', len(final_candidates))
	print(final_candidates)
	print('----------------------------')
	print('TRUE AS PAIRS: ', len(true_as_pairs))
	print(true_as_pairs)
	# generate predicted and true label array for f1 score
	y_true = []
	y_pred = []
	num_aspects_not_found = 0
	for true_as in true_as_pairs:
		y_true.append(true_as[1])
		final_candidates_a = list(map(lambda x: x[0], final_candidates))
		if true_as[0] in final_candidates_a: # aspect is in predicted candidates
			pred_as_index = final_candidates_a.index(true_as[0])
			y_pred.append(final_candidates[pred_as_index][1])
			pred_as = final_candidates.pop(pred_as_index)
			print('MATCH: true: ', true_as, ' pred: ', pred_as)
		else: # we purposely chose predicted label that doesn't match up to true label
			print('NO MATCH: true: ', true_as)
			num_aspects_not_found += 1
			if true_as[1] == candidate_labels[0]:
				y_pred.append(candidate_labels[1])
			elif true_as[1] == candidate_labels[1]:
				y_pred.append(candidate_labels[2])
			else:
				y_pred.append(candidate_labels[0])

	print('y_true: ', y_true)
	print('y_pred: ', y_pred)
	f1score = f1_score(y_true, y_pred, labels=candidate_labels, average='micro')
	print('f1 score: ', f1score)
	print('num aspects not found: ', num_aspects_not_found)
	print('num true as pairs: ', len(true_as_pairs))
	print('num candidate labels not used: ', len(final_candidates))


def results_parser(file, type_string):
	df= pd.read_csv(file)
	final_num_overlap = 0
	final_num_as1_prev = 0
	final_num_as1 = 0
	final_num_as2_prev = 0
	final_num_as2 = 0

	for i in range(df.shape[0]):
		oracle_summary = df['oracle'][i]
		oracle_summary_as = get_aspect_sentiment_pairs_only(oracle_summary)

		generated_summary = df['generated_summary'][i]
		if type_string == 'CONTENT PLANNING':
			index_to_slice = generated_summary.index('[SUMMARY]') + 10
			generated_summary = generated_summary[index_to_slice:]
		generated_summary_as = get_aspect_sentiment_pairs_only(generated_summary)

		num_as1_prev = len(oracle_summary_as)
		num_as2_prev = len(generated_summary_as)
		num_overlap, num_as1, num_as2 = aspect_sentiment_differences(oracle_summary_as, generated_summary_as)

		final_num_overlap += num_overlap
		final_num_as1_prev += num_as1_prev
		final_num_as1 += num_as1
		final_num_as2_prev += num_as2_prev
		final_num_as2 += num_as2

		
	print('SUMMARY FOR ', type_string,  ': ')
	print('num_overlap: ', final_num_overlap)
	print('num_as1_prev: ', final_num_as1_prev)
	print('num_as1: ', final_num_as1)
	print('num_as2_prev: ', final_num_as2_prev)
	print('num_as2: ', final_num_as2)
	print('-------------------------')


def content_planning_eval(file):
	# values of interest
	num_as_pairs = 0
	num_aspects_in_summary = 0
	num_as_in_summary = 0

	df= pd.read_csv(file)
	for i in range(df.shape[0]):
		generated_summary = df['generated_summary'][i]

		# get aspect sentiment pairs from aspect chain
		index_first = 14
		index_second = generated_summary.index('[SUMMARY]') - 1
		aspect_chain = generated_summary[index_first:index_second]
		as_pairs = aspect_chain.split(' | ')
		as_pairs_final = []
		for j in range(len(as_pairs)):
			seperator_index = as_pairs[j].index(',')
			aspect = as_pairs[j][:seperator_index]
			sentiment = as_pairs[j][seperator_index+1:]
			as_pairs_final.append((aspect, sentiment))

		# get generated summary without aspect chain
		index_to_slice = generated_summary.index('[SUMMARY]') + 10
		generated_summary_final = generated_summary[index_to_slice:]

		num_as_pairs += len(as_pairs_final)
		# count how many aspects is in generated summary
		for j in range(len(as_pairs_final)):
			singular_a = inflect.singular_noun(as_pairs_final[j][1])
			plural_a = inflect.plural_noun(as_pairs_final[j][1])
			if as_pairs_final[j][1] in generated_summary_final or (isinstance(singular_a, str) and singular_a in generated_summary_final) or (isinstance(plural_a, str) and plural_a in generated_summary_final):
				num_aspects_in_summary += 1

		# count how many as pairs are in generated summary
		as_pairs_final = convert_sentiment_words_to_labels(as_pairs_final)
		candidates = get_aspect_sentiment_candidates_1(generated_summary_final)
		final_candidates = convert_sentiment_words_to_labels(candidates)
		for j in range(len(as_pairs_final)):
			singular_as = (inflect.singular_noun(as_pairs_final[j][0]), as_pairs_final[j][1])
			plural_as = (inflect.plural_noun(as_pairs_final[j][0]), as_pairs_final[j][1])

			if as_pairs_final[j] in final_candidates or singular_as in final_candidates or plural_as in final_candidates:
				num_as_in_summary += 1
			else:
				print('not found: ', as_pairs_final[j])
				print('final candidates: ', final_candidates)
		

	print('SUMMARY FOR CONTENT PLANNING EVAL')
	print('num_as_pairs: ', num_as_pairs)
	print('num_aspects_in_summary: ', num_aspects_in_summary)
	print('num_as_in_summary: ', num_as_in_summary)


# generate_data('valid')
# generate_data('test')

# content_plan_file = 'yelp_aspect_sentiment_content_planning.csv'
# vanilla_file = 'yelp_test_vanilla_bart.csv'
# results_parser(content_plan_file, 'CONTENT PLANNING')
# results_parser(vanilla_file, 'VANILLA')

# content_plan_file = 'amazon_aspect_sentiment_content_planning.csv'
# vanilla_file = 'amazon_test_finetune_bart_large.csv'
# results_parser(content_plan_file, 'CONTENT PLANNING')
# results_parser(vanilla_file, 'VANILLA')

# content_planning_eval('yelp_aspect_sentiment_content_planning.csv')
# content_planning_eval('amazon_aspect_sentiment_content_planning.csv')