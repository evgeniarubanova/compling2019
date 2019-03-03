import os, re
from string import punctuation
import numpy as np
import json
from collections import Counter
from pprint import pprint
punct = set(punctuation)
import textdistance
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from difflib import get_close_matches

corpus = [sent.split() for sent in open('corpus_ng.txt', encoding='utf8').read().splitlines()] 
WORDS = Counter()
for sent in corpus:
    WORDS.update(sent)

vocab = list(WORDS.keys())
id2word = {i:word for i, word in enumerate(vocab)}

vec = TfidfVectorizer(analyzer='char', ngram_range=(1,1))
X = vec.fit_transform(vocab)

def get_closest_hybrid_match(text, X, vec, metric=textdistance.levenshtein):
    v = vec.transform([text])
    similarities = cosine_distances(v, X)
    topn = similarities.argsort()[0][:5]
    variants =[id2word[top] for top in topn]
    similarities = Counter()
    for word in variants:
        similarities[word] = metric.normalized_similarity(text, word) 
    return similarities.most_common(1)[0][0]

get_closest_hybrid_match('патаму', X, vec)

bad = open('sents_with_mistakes.txt', encoding='utf8').read().splitlines()
true = open('correct_sents.txt', encoding='utf8').read().splitlines()

def align_words(sent_1, sent_2):
    tokens_1 = sent_1.lower().split()
    tokens_2 = sent_2.lower().split()
    tokens_1 = [re.sub('(^\W+|\W+$)', '', token) for token in tokens_1 if (set(token)-punct)]
    tokens_2 = [re.sub('(^\W+|\W+$)', '', token) for token in tokens_2 if (set(token)-punct)]
    return list(zip(tokens_1, tokens_2))

arr = []
correct = 0
total = 0

for i in range(len(true)):
    word_pairs = align_words(true[i], bad[i])
    for pair in word_pairs:
        predicted = get_closest_hybrid_match(pair[1], X, vec)
        if predicted == pair[0]:
            correct += 1
        else:
            arr.append([pair, predicted])
            print([pair, predicted])
        total += 1
    if not i % 10:
        print(i)
        print(correct/total) 

