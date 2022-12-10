##author: Mohamed Elaraby
##Date: July 7th 2022

from nltk.tokenize import sent_tokenize
import pandas as pd
import argparse
import random
import numpy as np
import swifter


my_parser = argparse.ArgumentParser()
my_parser.add_argument("-reviews", type=str)
my_parser.add_argument("-dapt", type=str)
my_parser.add_argument("-mlm_prob", type=float)
args = my_parser.parse_args()


def text_infilling(sent, mask_probability=0.05, lamda=3):
    '''
    inputs:
        sent: a sentence string
        mask_probability: probability for masking tokens
        lamda: lamda for poission distribution
    outputs:
        sent: a list of tokens with masked tokens
    '''
    sent = sent.split()
    length = len(sent)
    mask_indices = (np.random.uniform(0, 1, length) < mask_probability) * 1
    span_list = np.random.poisson(lamda, length)  # lamda for poission distribution
    nonzero_idx = np.nonzero(mask_indices)[0]
    for item in nonzero_idx:
        span = min(span_list[item], 5)    # maximum mask 5 continuous tokens
        for i in range(span):
            if item+i >= length:
                continue
            mask_indices[item+i] = 1
    for i in range(length):
        if mask_indices[i] == 1:
            sent[i] = '<mask>'

    # merge the <mask>s to one <mask>
    final_sent = []
    mask_flag = 0
    for word in sent:
        if word != '<mask>':
            mask_flag = 0
            final_sent.append(word)
        else:
            if mask_flag == 0:
                final_sent.append(word)
            mask_flag = 1
    return final_sent

def sent_permutation(sent):
    '''
    inputs:
        sent: a sentence string
    outputs:
        shuffle_sent: a string after sentence permutations
    '''
    # split sentences based on '.'
    splits = sent_tokenize(sent)
    random.shuffle(splits)

    return " ".join(splits)


def add_noise(sent, mask_probability):
    noisy_sent = sent_permutation(sent)
    noisy_sent = text_infilling(noisy_sent, mask_probability)
    noisy_sent = " ".join(noisy_sent)
    return noisy_sent


df_reviews = pd.read_csv(args.reviews)
df_reviews['dapt_reviews'] = df_reviews.swifter.apply(lambda row: add_noise(row['reviews'], mask_probability=args.mlm_prob), axis=1)

df_reviews.to_csv(args.dapt, index=False)