import pandas as pd
import gensim
import numpy as np
import math
import io
from numpy import dot
from numpy.linalg import norm

def cosineSimilarity(a,b):
  return dot(a, b)/(norm(a)*norm(b))

DATA = "../data/semevalSimilarity/test/subtask1-monolingual/data/"
OUTPUT = "../data/semevalSimilarity/test/subtask1-monolingual/keys/"


#Download https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
def w2vec_model():
  print "LOADING WORD2VEC MODEL"
  model = gensim.models.Word2Vec.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin', binary=True)
  print "LOADED WORD2VEC MODEL"
  return model

def accuracy_averaged(result, GoldOutput):
  avg_error = 0
  cnt = len(result)
  for ind in range(len(result)):
    avg_error += abs(result[ind] - GoldOutput[ind])
  print avg_error/cnt
  return avg_error/cnt

def vector(model, word):
  if word in model:
    return model[word]
  else:
    return model['UNK']

def monolingualSimilarity(word2VecModelFile, language='en'):
  model = w2vec_model()
  filename = language + ".test.data.txt"
  f = io.open(DATA+filename, encoding="utf-8")
  p_data = pd.read_csv(f, sep='\t')
  Output = []
  for i in xrange(len(p_data)):
    Output.append(4.0 * cosineSimilarity(vector(model, p_data['word1'][i]), vector(model, p_data['word2'][i])))

  filename = language + ".test.gold.txt"
  f = io.open(OUTPUT+filename, encoding="utf-8")
  gold_output_data = pd.read_csv(f, sep='\t')
  accuracy_averaged(Output, gold_output_data['Output'])

monolingualSimilarity(None)