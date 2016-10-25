# Import libraries

import json
import os
import sys
import numpy as np
import copy
from gensim.models import doc2vec
from gensim.models import Word2Vec
from collections import namedtuple
from scipy.spatial import distance
import re


sentence = ''
doc1 = []
wordvector = {}
phrasevector = {}
phrasewordvector = {}
paragraphvector = {}
paragraphwordvector = {}
distancevector = []
dictionarysize = 0

# Load data
def LoadData():
    f = open('workfile.txt', 'r')
    dictionarywords = []
    for line in f:
        global sentence
        sentence = line
        trimmedline = line.strip()
        phrases = trimmedline.split(".")
        for phrase in phrases:
            trimmedphrase = phrase.strip()
            if trimmedphrase != '':
                doc1.append(trimmedphrase)
                words = trimmedphrase.split(" ")
                dictionarywords.extend(words)
    global dictionarysize
    dictionarysize = len(set(dictionarywords))

# Transform data (you can add more data preprocessing steps) 
def PhraseEmbedding():
    docs = []
    analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
    for i, text in enumerate(doc1):
        words = text.lower().split()
        tags = [i]
        docs.append(analyzedDocument(words, tags))
    # Train model (set min_count = 1, if you want the model to work with the provided example data set)
    global dictionarysize
    model = doc2vec.Doc2Vec(docs, size = dictionarysize, window = 300, min_count = 1, workers = 4)
    i = 0
    item = {}
    for phrases in model.docvecs:
        phrase = " ".join(docs[i].words)
        item['phrase'] = phrase
        item['vector'] = model.docvecs[i]
        item['index'] = i
        i = i + 1
        phrasevector[phrase] = copy.deepcopy(item)
    i = 0
    item = {}
    for word in model.index2word:
        item['word'] = word
        item['index'] = i
        item['vector'] = model.syn0[i]
        i = i + 1
        phrasewordvector[word] = copy.deepcopy(item)
    return model


# Transform data (you can add more data preprocessing steps) 
def ParagraphEmbedding():
    docs = []
    global sentence
    analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
    sen = sentence.lower()
    words = re.split("\s+|\s*,\s*|\s*\.\s*|\s+", sen)
    tags = [0]
    docs.append(analyzedDocument(words, tags))
    # Train model (set min_count = 1, if you want the model to work with the provided example data set)
    global dictionarysize
    print dictionarysize
    model = doc2vec.Doc2Vec(docs, size = dictionarysize, window = 300, min_count = 1, workers = 4)
    i = 0
    item = {}
    for phrases in model.docvecs:
        phrase = " ".join(docs[i].words)
        item['phrase'] = phrase
        item['vector'] = model.docvecs[i]
        item['index'] = i
        i = i + 1
        paragraphvector[phrase] = copy.deepcopy(item)
    i = 0
    item = {}
    for word in model.index2word:
        item['word'] = word
        item['index'] = i
        item['vector'] = model.syn0[i]
        i = i + 1
        paragraphwordvector[word] = copy.deepcopy(item)
    return model

def WordEmbedding():
    sentences = doc1
    vocab = [s.encode('utf-8').split() for s in sentences]
    model = Word2Vec(vocab, size=dictionarysize, window=300, min_count=1, workers=4)
    item = {}
    i = 0;
    for (k,v) in model.vocab.items():
        item['word'] = k
        item['count'] = v.count
        item['vector'] = model.syn0[i]
        i = i + 1
        item['index'] = v.index
        wordvector[k] = copy.deepcopy(item)
    return model

def KNearestPhrases(k, a):
    for word in wordvector:
        b = np.array(wordvector[word]['vector'])
        dist = distance.euclidean(a,b)
        distancevector.append({'word' : word, 'dist' : dist})
    newdistvector = sorted(distancevector, key=lambda dictionary: dictionary['dist'])
    for num in range(0,k):
        print newdistvector[num]['word'] 

LoadData()
phrasemodel = PhraseEmbedding()
wordmodel = WordEmbedding()
paramodel = ParagraphEmbedding()
a = np.random.rand(dictionarysize)
KNearestPhrases(3, a)

# Get the vectors
#weights = phrasemodel.syn0
#np.save(open('/Users/rameshkumar/Desktop/nlp/data/weights', 'wb'), weights)
#vocab = dict([(k, v.index) for k, v in phrasemodel.vocab.items()])
#with open('/Users/rameshkumar/Desktop/nlp/data/vocab.txt', 'w') as f:
#    f.write(json.dumps(vocab))

