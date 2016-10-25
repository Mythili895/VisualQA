from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import recurrent
from keras.layers import Dense, Merge, Dropout, RepeatVector
import numpy as np
import itertools
import json

np.random.seed(1337)

class image_data(object):
    def __init__(self, phrases=[], quest=None, ans=None):
        self._phrases = phrases
        self._questions = quest
        self._answers = ans

    @property
    def Phrases(self):
        return self._phrases

    @property
    def Questions(self):
        return self._questions

    @property
    def Answers(self):
        return self._answers


def get_training_data(size=50):
    listImageData = []
    qafile = open('/Users/rameshkumar/Desktop/nlp/data/question_answers.json')
    qa_data = json.load(qafile)
    qafile.close()

    regionfile = open('/Users/rameshkumar/Desktop/nlp/data/region_descriptions.json')
    region_data = json.load(regionfile)
    regionfile.close()

    count = 0

    for i in range(0, len(region_data)):
        listQ = []
        listA = []
        listP = []
        qas = qa_data[i]['qas']
        regions = region_data[i]['regions']
        if len(qas) == 0:
            continue
        for qas_dict in qas:
            listQ.append(qas_dict['question'].strip())
            listA.append(qas_dict['answer'].strip())
        for reg in regions:
            listP.append(reg['phrase'].strip())

        for i in xrange(len(listQ)):
            newImage_data = image_data(listP, listQ[i], listA[i])
            listImageData.append(newImage_data)
            count += 1
            if count == size:
                break
        if count == size:
            break

    return listImageData


def get_vocab(listDetails):
    listWords = []
    prevPhr = None
    for detail in listDetails:
        if prevPhr != detail.Phrases:
            for phr in detail.Phrases:
                listWords.append(tokenize(phr))
            prevPhr = detail.Phrases
        listWords.append(tokenize(detail.Questions))
        listWords.append(tokenize(detail.Answers))

    #return set(listWords)

    return set(itertools.chain(*listWords))

def tokenize(sent):
    lemma = lambda x: x.strip().lower().split(' ')
    #return [x.strip().lower() for x in re.split('(\W+)?', sent) if x.strip()]
    return lemma(sent)

def get_maxLengths(listDetails):
    max_phraseslen = 0
    max_questionslen = 0
    max_answerslen = 0
    prev_phrase = None
    count = 0

    for lis in listDetails:
        count += 1
        if prev_phrase != lis.Phrases:
            for phrase in lis.Phrases:
                leng = len(tokenize(phrase))
                if  leng > max_phraseslen:
                    max_phraseslen = leng
            prev_phrase = lis.Phrases

        leng = len(tokenize(lis.Questions))
        if leng > max_questionslen:
            max_questionslen = leng

        leng = len(tokenize(lis.Answers))
        if leng > max_answerslen:
            max_answerslen = leng

    return (max_phraseslen, max_questionslen, max_answerslen)

def vectorize_stories(listDetails, word_idx, phrase_length, question_length):
    phrases = []
    questions = []
    answers = []

    for lis in listDetails:
        lisWords = []
        for phr in lis.Phrases:
            lisWords.extend(tokenize(phr))
        x = [word_idx[w] for w in lisWords]
        phrases.append(x)
        xq = [word_idx[w] for w in tokenize(lis.Questions)]
        questions.append(xq)
        y = np.zeros(len(word_idx) + 1)
        wrd = tokenize(lis.Answers)
        for w in wrd:
            y[word_idx[w]] = 1
        answers.append(y)

    return pad_sequences(phrases, maxlen=phrase_length), pad_sequences(questions, maxlen=question_length), np.array(answers)


DATASET_SIZE = 10
EMBED_HIDDEN_SIZE = 40
RNN = recurrent.LSTM
BATCH_SIZE = 32
EPOCHS = 10
PHRASES_HIDDEN_SIZE = 256
QUESTION_HIDDEN_SIZE = 128

listImageDetails = get_training_data(DATASET_SIZE)

vocab_words = get_vocab(listImageDetails)

vocab_size = len(vocab_words) + 1
word_idx = dict((v, i) for i, v in enumerate(vocab_words))
(max_phrase_length, max_question_length, max_answer_length) = get_maxLengths(listImageDetails)

Phrase_train, Question_train, Answer_train = vectorize_stories(listImageDetails[:DATASET_SIZE/2], word_idx, max_phrase_length, max_question_length)
Phrase_test, Question_test, Answer_test = vectorize_stories(listImageDetails[DATASET_SIZE/2:], word_idx, max_phrase_length, max_question_length)

phrasRNN = Sequential()
phrasRNN.add(Embedding(vocab_size, EMBED_HIDDEN_SIZE, input_length=max_phrase_length))
phrasRNN.add(Dropout(0.3))

quesRNN = Sequential()
quesRNN.add(Embedding(vocab_size, EMBED_HIDDEN_SIZE, input_length=max_question_length))
quesRNN.add(Dropout(0.3))
quesRNN.add(RNN(EMBED_HIDDEN_SIZE, return_sequences=False))
quesRNN.add(RepeatVector(max_phrase_length))

model = Sequential()
model.add(Merge([phrasRNN, quesRNN], mode='concat'))
model.add(RNN(EMBED_HIDDEN_SIZE, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(vocab_size, activation='tanh'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit([Phrase_train, Question_train], Answer_train, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, validation_split=0.05)
loss, acc = model.evaluate([Phrase_test, Question_test], Answer_test, batch_size=BATCH_SIZE)
print 'LOSS=', loss, 'ACCURACY=', acc
