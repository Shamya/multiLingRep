import pandas as pd
import gensim
import numpy as np
import math
import io
from numpy import dot
from numpy.linalg import norm


def cosineSimilarity(a,b):
  if(norm(a) == 0 or norm(b) == 0):
    return 0
  return dot(a, b)/(norm(a)*norm(b))

DATA = "../data/semevalSimilarity/test/subtask2-crosslingual/data/"
OUTPUT = "../data/semevalSimilarity/test/subtask2-crosslingual/keys/"


#Download https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
def w2vec_model(lang='en'):
  hashVal = {"en" : "eng", "es" : "esp", "mul" : "mul"}
  print "LOADING WORD2VEC MODEL"
  model = gensim.models.Word2Vec.load("../data/word_vectors_" + hashVal[lang] + ".txt")
  # model = gensim.models.Word2Vec.load_word2vec_format('../data/word_vectors_eng.txt', binary=False)
  print "LOADED WORD2VEC MODEL"
  return model

def accuracy_averaged(result, GoldOutput):
  avg_error = 0
  cnt = len(result)
  for ind in range(len(result)):
    avg_error += (abs(result[ind] - GoldOutput[ind]))
  print avg_error/cnt
  return avg_error/cnt

def vector(model, word):
  if word in model:
    return model[word]
  else:
    return np.zeros(300)

def monolingualSimilarity(word2VecModelFile, language='mul'):
  model = w2vec_model(language)
  testfile = "en-es.test.data.txt"
  outputfile = "en-es.test.gold.txt"
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
  accuracy_averaged(Output, gold_output_data['Output'])

monolingualSimilarity(None)