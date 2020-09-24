# -*- coding: utf-8 -*-

""" Use torchMoji to score texts for emoji distribution.

The resulting emoji ids (0-63) correspond to the mapping
in emoji_overview.png file at the root of the torchMoji repo.

Writes the result to a csv file.
"""
from __future__ import print_function, division, unicode_literals
# import example_helper
import json
import csv
import numpy as np
from collections import Counter 

from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchmoji.model_def import torchmoji_emojis
from torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH

# OUTPUT_PATH = 'test_sentences.csv'

# TEST_SENTENCES = ['I love mom\'s cooking',
#                   'I love how you never reply back..',
#                   'I love cruising with my homies',
#                   'I love messing with yo mind!!',
#                   'I love you and now you\'re just gone..',
#                   'This is shit',
#                   'This is the shit']


def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]

def most_frequent(List): 
    return max(set(List), key = List.count) 

with open(VOCAB_PATH, 'r') as f:
    vocabulary = json.load(f)

maxlen = 30
st = SentenceTokenizer(vocabulary, maxlen)

model = torchmoji_emojis(PRETRAINED_PATH)

def get_emoji_score(sentence, label):
    mapper = {"angry":anger, "disgusted":disgust, "terrified":fear, "joyful":joy, "sad":sad, "surprised":surprise}
    tokenized, _, _ = st.tokenize_sentences(sentence)
    prob = model(tokenized)
    scores = []
    acc = []
    for i, t in enumerate(sentence):
        t_prob = prob[i]
        ind_top = top_elements(t_prob, 2)
        if(emoji_list[ind_top[0]][1] in mapper[label] or emoji_list[ind_top[1]][1] in mapper[label]): 
            acc.append(1)
        else:
            acc.append(0)
        
        scores.append([emoji_list[i][1].upper() for i in ind_top])
        
    counter = Counter(sum(scores,[]))
    
    return "".join([ f'({c})' + r'{\NotoEmoji\symbol{"' +str(w)+'}}' for w, c in counter.most_common(4)]),np.mean(acc)


joy = ["1f602", "1f604","1f60a","1f60b", "1f60c", "1f60d", "1f60e", "1f60f", "263a", "1f618",
        "1f61c","2764", "1f496", "1f495", "1f601","2665","270c","2661","1f3a7","1f49c","1f496","1f499"]
sad = ["1f614", "1f615", "1f62b", "1f629", "1f622", 
       "1f62a", "1f62d", "1f494"]
anger= ["1f62c", "1f620", "1f610","1f611", "1f621", "1f616", "1f624"]
disgust = ["1f637"]
fear = ["1f605"]
surprise = ["1f633"]


emoji_list = [["\U0001f602","1f602"],
            ["\U0001f612","1f612"],
            ["\U0001f629","1f629"],
            ["\U0001f62d","1f62d"],
            ["\U0001f60d","1f60d"],
            ["\U0001f614","1f614"],
            ["\U0001f44c","1f44c"],
            ["\U0001f60a","1f60a"],
            ["\u2764","2764"],
            ["\U0001f60f","1f60f"],
            ["\U0001f601","1f601"],
            ["\U0001f3b6","1f3b6"],
            ["\U0001f633","1f633"],
            ["\U0001f4af","1f4af"],
            ["\U0001f634","1f634"],
            ["\U0001f60c","1f60c"],
            ["\u263a","263a"],
            ["\U0001f64c","1f64c"],
            ["\U0001f495","1f495"],
            ["\U0001f611","1f611"],
            ["\U0001f605","1f605"],
            ["\U0001f64f","1f64f"],
            ["\U0001f615","1f615"],
            ["\U0001f618","1f618"],
            ["\u2665","2665"],
            ["\U0001f610","1f610"],
            ["\U0001f481","1f481"],
            ["\U0001f61e","1f61e"],
            ["\U0001f648","1f648"],
            ["\U0001f62b","1f62b"],
            ["\u270c","270c"],
            ["\U0001f60e","1f60e"],
            ["\U0001f621","1f621"],
            ["\U0001f44d","1f44d"],
            ["\U0001f622","1f622"],
            ["\U0001f62a","1f62a"],
            ["\U0001f60b","1f60b"],
            ["\U0001f624","1f624"],
            ["\u270b","270b"],
            ["\U0001f637","1f637"],
            ["\U0001f44f","1f44f"],
            ["\U0001f440","1f440"],
            ["\U0001f52b","1f52b"],
            ["\U0001f623","1f623"],
            ["\U0001f608","1f608"],
            ["\U0001f613","1f613"],
            ["\U0001f494","1f494"],
            ["\u2661","2661"],
            ["\U0001f3a7","1f3a7"],
            ["\U0001f64a","1f64a"],
            ["\U0001f609","1f609"],
            ["\U0001f480","1f480"],
            ["\U0001f616","1f616"],
            ["\U0001f604","1f604"],
            ["\U0001f61c","1f61c"],
            ["\U0001f620","1f620"],
            ["\U0001f645","1f645"],
            ["\U0001f4aa","1f4aa"],
            ["\U0001f44a","1f44a"],
            ["\U0001f49c","1f49c"],
            ["\U0001f496","1f496"],
            ["\U0001f499","1f499"],
            ["\U0001f62c","1f62c"],
            ["\u2728","2728"]]