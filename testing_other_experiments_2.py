import pandas as pd
import csv
import numpy as np
import gensim
from libraries.tweetUtilities import tweetTokenizer, tweetPreProcess
from libraries.sentiment_classifier import AvgVector, vector, classifyusingAvgVectors, w2vec_model, classify, sentimentAnalysisSpanishDataset
import math
import io
from numpy import dot
from numpy.linalg import norm
from libraries.fileUtilities import load_pkl_file
from numpy import array


def cosineSimilarity(a,b):
  if(norm(a) == 0 or norm(b) == 0):
    return 0
  return dot(a, b)/(norm(a)*norm(b))

DATA = "data/semevalSimilarity/test/"
OUTPUT = "data/semevalSimilarity/test/"


def accuracy_averaged(result, GoldOutput):
  avg_error = 0
  cnt = len(result)
  for ind in range(len(result)):
    avg_error += (abs(result[ind] - GoldOutput[ind]))
  print avg_error/cnt
  return avg_error/cnt

def vector(model, word):
  if word in model:
    return array(model[word])
  else:
    return np.zeros(300)

def Similarity(model, testfile, goldstandardfile):
    # model = w2vec_model(language)
    outputfile = goldstandardfile
    filename =  testfile
    f = io.open(DATA+filename, encoding="utf-8")
    p_data = pd.read_csv(f, sep='\t')
    Output = []
    for i in xrange(len(p_data)):
        result = 4.0 * cosineSimilarity(vector(model, p_data['word1'][i].lower()), vector(model, p_data['word2'][i].lower()))
        print p_data['word1'][i], p_data['word2'][i], result
        Output.append(result)
    filename = outputfile
    f = io.open(OUTPUT+filename, encoding="utf-8")
    gold_output_data = pd.read_csv(f, sep='\t')

  # Similarity(en_es_model, "subtask2-crosslingual/data/en-es.test.data.txt", "subtask2-crosslingual/keys/en-es.test.gold.txt");

def w2vec_model(filename):
  print "LOADING WORD2VEC MODEL - " + filename
  model = gensim.models.Word2Vec.load("data/" + filename)
  # model = gensim.models.Word2Vec.load_word2vec_format('../data/word_vectors_eng.txt', binary=False)
  print "LOADED WORD2VEC MODEL - " + filename
  return model

def load_factorie_model(filename):
  model = load_pkl_file(filename)
  return model

def sentimentAnalysisEnglishDataset():
    FILE = 'data/EnglishSentimentDataSet/English_sentiment_twitter.tsv'
    X = []
    Y = []
    with open(FILE, 'rU') as csvfile:
        spamreader = csv.reader(csvfile, dialect=csv.excel_tab, delimiter='\t')
        for row in spamreader:
            if( len(row[1:]) > 1):
                Tweet = ' '.join(row[1:])
                if(Tweet.strip()=='Not Available'):
                    continue
            else:
                if(row[1]=='Not Available'):
                    continue
                Tweet = row[1]
            print Tweet
            print Tweet.strip()
            if(row[0]=='positive'):
                Y.append(1)
            elif(row[0]=='neutral'):
                Y.append(0)
            elif(row[0]=='negative'):
                Y.append(-1)
            else:
                assert False
            X.append(tweetPreProcess(Tweet,'english'));
    return np.array(X), Y


def sentimentAnalysisItalianDataset():
    FILE = 'data/ItalianSentimentDataSet/italian_sentiment_twitter.csv'
    X = []
    Y = []
    with open(FILE, 'rU') as csvfile:
        spamreader = csv.reader(csvfile, dialect=csv.excel_tab, delimiter=',')
        for row in spamreader:

            if( len(row[2:]) > 1):
                Tweet = ' '.join(row[2:])
                if(Tweet.strip()=='Not Available'):
                    continue
            else:
                if(row[2]=='Not Available'):
                    continue
                Tweet = row[2]
            if(row[0]=='0'):
                Y.append(0)
            elif(row[0]=='1'):
                Y.append(1)
            else:
                assert False
            t = tweetPreProcess(Tweet,'italian')
            print t
            X.append(t);
    return np.array(X), Y

X,Y = sentimentAnalysisItalianDataset()
indices_pos = [i for i, e in enumerate(Y) if e == 1]
indices_neg = [i for i, e in enumerate(Y) if e == 0]


neg_len = len(indices_neg)
pos_len = len(indices_pos)



train_X_neg  =  X[indices_neg][:neg_len/2]
train_X_pos  =  X[indices_pos][:pos_len/2]


train_Y_neg  = np.full((len(train_X_neg)), 0)
train_Y_pos  = np.full((len(train_X_pos)), 1)

test_X_neg  =  X[indices_neg][neg_len/2:]
test_X_pos  =  X[indices_pos][pos_len/2:]


test_Y_neg  = np.full((len(test_X_neg)), 0)
test_Y_pos  = np.full((len(test_X_pos)), 1)
train_X = np.concatenate((train_X_neg,train_X_pos), axis=0)
train_Y = np.concatenate((train_Y_neg,train_Y_pos), axis=0)
test_X = np.concatenate((test_X_neg,test_X_pos), axis=0)
test_Y = np.concatenate((test_Y_neg,test_Y_pos), axis=0)
entire_italian_X = np.concatenate((train_X,test_X), axis=0)
entire_italian_Y = np.concatenate((train_Y,test_Y), axis=0)


train_italian_df = pd.DataFrame({'Sentence':(train_X),'ClassifiedOutput':(train_Y)})
test_italian_df =  pd.DataFrame({'Sentence':(test_X),'ClassifiedOutput':(test_Y)})
entire_italian_df = pd.DataFrame({'Sentence':(entire_italian_X),'ClassifiedOutput':(entire_italian_Y)})


X,Y = sentimentAnalysisEnglishDataset()
indices_pos = [i for i, e in enumerate(Y) if e == 1]
indices_neg = [i for i, e in enumerate(Y) if e == -1]
indices_neu = [i for i, e in enumerate(Y) if e == 0]
neu_len = len(indices_neu)
neg_len = len(indices_neg)
pos_len = len(indices_pos)


train_X_neu  =  X[indices_neu][:neu_len/2]
train_X_neg  =  X[indices_neg][:neg_len/2]
train_X_pos  =  X[indices_pos][:pos_len/2]

train_Y_neu  = np.full((len(train_X_neu)), 0)
train_Y_neg  = np.full((len(train_X_neg)), 0)
train_Y_pos  = np.full((len(train_X_pos)), 1)

test_X_neu  =  X[indices_neu][neu_len/2:]
test_X_neg  =  X[indices_neg][neg_len/2:]
test_X_pos  =  X[indices_pos][pos_len/2:]


test_Y_neu  = np.full((len(test_X_neu)), 0)
test_Y_neg  = np.full((len(test_X_neg)), 0)
test_Y_pos  = np.full((len(test_X_pos)), 1)


train_X = np.concatenate((train_X_neg,train_X_pos), axis=0)
train_Y = np.concatenate((train_Y_neg,train_Y_pos), axis=0)
test_X = np.concatenate((test_X_neg,test_X_pos), axis=0)
test_Y = np.concatenate((test_Y_neg,test_Y_pos), axis=0)
entire_english_X = np.concatenate((train_X,test_X), axis=0)
entire_english_Y = np.concatenate((train_Y,test_Y), axis=0)


train_english_df = pd.DataFrame({'Sentence':(train_X),'ClassifiedOutput':(train_Y)})
test_english_df =  pd.DataFrame({'Sentence':(test_X),'ClassifiedOutput':(test_Y)})
entire_english_df = pd.DataFrame({'Sentence':(entire_english_X),'ClassifiedOutput':(entire_english_Y)})


train_spanish_df, test_spanish_df, entire_spanish_df = sentimentAnalysisSpanishDataset()

print "USING ES EN MODEL"
en_es_model = w2vec_model('wv_enes_mul.txt')

print "Trained on Es , Tested on En -EXP # 10"
train_df = train_spanish_df
test_df = test_english_df
print "Length of Train : ", str(len(train_df))
print "Length of Test : ", str(len(test_df))
classifyusingAvgVectors(train_df,test_df,dimensionOfVector=300, model=en_es_model)

print "Trained on English , Tested on Spanish-EXP # 16"
train_df = train_english_df
test_df = test_spanish_df
print "Length of Train : ", str(len(train_df))
print "Length of Test : ", str(len(test_df))
classifyusingAvgVectors(train_df,test_df,dimensionOfVector=300, model=en_es_model)

es_en_it_train_df = pd.concat([train_english_df, train_spanish_df, train_italian_df], ignore_index=True)
es_en_it_test_df = pd.concat([test_english_df, test_spanish_df, test_italian_df], ignore_index=True)
es_en_it_entire_df = pd.concat([entire_english_df, entire_spanish_df, entire_italian_df], ignore_index=True)


print "USING ACS **********************************************************"

# USING ACS GETTING RESULTS :
en_es_model = w2vec_model('wv_enes_acs.txt')

print "Trained on Es , Tested on En -EXP # 2"
train_df = train_spanish_df
test_df = test_english_df
print "Length of Train : ", str(len(train_df))
print "Length of Test : ", str(len(test_df))
classifyusingAvgVectors(train_df,test_df,dimensionOfVector=300, model=en_es_model)

print "Trained on English , Tested on Spanish-EXP # 18"
train_df = train_english_df
test_df = test_spanish_df
print "Length of Train : ", str(len(train_df))
print "Length of Test : ", str(len(test_df))
classifyusingAvgVectors(train_df,test_df,dimensionOfVector=300, model=en_es_model)

print "Other Experiments ****************************************************"

en_es_model = w2vec_model('wv_enit_mul.txt')

print "EN IT Model"

print "Trained on It , Tested on It -EXP # 47"
train_df = train_italian_df
test_df = test_italian_df
print "Length of Train : ", str(len(train_df))
print "Length of Test : ", str(len(test_df))
classifyusingAvgVectors(train_df,test_df,dimensionOfVector=300, model=en_es_model)


en_it_train_df = pd.concat([train_english_df, train_italian_df], ignore_index=True)

print "Trained on En + It , Tested on It -EXP # 50"
train_df = en_it_train_df
test_df = test_italian_df
print "Length of Train : ", str(len(train_df))
print "Length of Test : ", str(len(test_df))
classifyusingAvgVectors(train_df,test_df,dimensionOfVector=300, model=en_es_model)



en_es_model = w2vec_model('word_vectors_ita.txt')

print "IT Model"

print "Trained on It , Tested on It -EXP # 46"
train_df = train_italian_df
test_df = test_italian_df
print "Length of Train : ", str(len(train_df))
print "Length of Test : ", str(len(test_df))
classifyusingAvgVectors(train_df,test_df,dimensionOfVector=300, model=en_es_model)


