from xml.dom.minidom import parse as pr
import pandas as pd
from libraries.tweetUtilities import tweetTokenizer, tweetPreProcess
from libraries.dataSetUtility import classify
from libraries.MLClassifier import classifierModel, RandForest, LR
from libraries.evaluationMetrics import print_accuracy_fscore
import numpy as np
import gensim

TRAIN_FILES = ['data/SpanishSentimentDataSet/Train/general-tweets-train-tagged.xml']
TEST_FILES = [ 'data/SpanishSentimentDataSet/Train/politics2013-tweets-test-tagged.xml' ]
def tweetsFromFile(filename):
  tree = pr(filename)
  tweets_tree = tree.documentElement
  tweets = tweets_tree.getElementsByTagName("tweet")
  return tweets

def sentimentData(files):
  data = []
  for filename in files:
    print filename
    tweets = tweetsFromFile(filename)
    for tweet in tweets:
        try:
            content = tweet.getElementsByTagName("content")[0].childNodes[0].data
        except:
            continue
        polarity = tweet.getElementsByTagName("sentiments")[0].getElementsByTagName("value")[0].childNodes[0].data
        positive = (polarity == "P" or polarity == "P+")
        negative = (polarity == "N" or polarity=="N+")
        if(positive or negative):
          if(positive):
            value = 1
          else:
            value = 0
          data.append([tweetPreProcess(content,'spanish'), value])
  return data


#Download https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
def w2vec_model(lang='en'):
  hashVal = {"en" : "eng", "es" : "esp", "mul" : "mul"}
  print "LOADING WORD2VEC MODEL"
  model = gensim.models.Word2Vec.load("data/word_vectors_" + hashVal[lang] + ".txt")
  # model = gensim.models.Word2Vec.load_word2vec_format('../data/word_vectors_eng.txt', binary=False)
  print "LOADED WORD2VEC MODEL"
  return model

def vector(model, word, dimensionOfVector):
  if word in model:
    return model[word]
  else:
    return np.zeros(dimensionOfVector)

def AvgVector(tokens, w2v_model, dimensionOfVector):
  Vector = np.zeros((dimensionOfVector,))
  for token in tokens:
    print token
    vec_temp = vector(w2v_model, token.lower(), dimensionOfVector)
    Vector += vec_temp - np.mean(vec_temp)
  if(len(tokens) > 0):
    return Vector/len(tokens)
  return np.zeros(dimensionOfVector)

def classifyusingAvgVectors(train_df,test_df,dimensionOfVector=300):
  w2v_model = w2vec_model('mul')
  X = []
  test_X = []
  for ind in xrange(len(train_df)):
    tokens = train_df['Sentence'][ind].split()
    X.append(AvgVector(tokens, w2v_model, dimensionOfVector))
  for ind in xrange(len(test_df)):
    tokens = test_df['Sentence'][ind].split()
    test_X.append(AvgVector(tokens, w2v_model, dimensionOfVector))

  Y = list(train_df['ClassifiedOutput'])
  test_Y = list(test_df['ClassifiedOutput'])
  classifier = classifierModel(LR)
  classifier = classifier.fit(np.asarray(X), list(Y))
  train_result = classifier.predict(X)
  test_result = classifier.predict(test_X)
  print_accuracy_fscore(train_result, list(Y))
  print_accuracy_fscore(test_result, list(test_Y))

  # classifier = classifierModel(LR)

  # classifier = classifier.fit(np.asarray(X), list(Y))
  # train_result = classifier.predict(X)
  # test_result = classifier.predict(test_X)
  # print_accuracy_fscore(train_result, list(train["ClassifiedOutput"]))
  # print_accuracy_fscore(test_result, list(test["ClassifiedOutput"]))




def sentimentAnalysisFrenchDataset():
  train_data = sentimentData(TRAIN_FILES)
  print len(train_data)
  test_data = sentimentData(TEST_FILES)
  print len(test_data)

  train_df = pd.DataFrame(train_data, columns = ['Sentence' , 'ClassifiedOutput'])
  test_df = pd.DataFrame(test_data, columns = ['Sentence' , 'ClassifiedOutput'])

  # classify(train_df, test_df)

  return classifyusingAvgVectors(train_df, test_df)

