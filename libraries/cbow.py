from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile
import nltk
from nltk.corpus import stopwords
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from libraries.fileUtilities import save_pkl_file

batch_size = 500
embedding_size = 200
vocabulary_size = 2000
generations = 50000
model_learning_rate = 0.001
num_sampled = int(batch_size/2)
window_size = 3
data_index = 0

def getWords(Sentences):
    words = []
    for sentence in Sentences:
        for word in sentence:
            words.append(word)
    return words

def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary



def generate_batch(data, batch_size, context_window):
  global data_index
  #assert batch_size % num_skips == 0
  #assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size, 2*context_window), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2*context_window + 1 
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size):
    batch[i, :] = [token for idx, token in enumerate(buffer) if idx != context_window]
    labels[i, 0] = buffer[context_window]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels

def generate_embeddings(Sentences):


    # Declare stop words
    stops = stopwords.words('english')
    

    words = getWords(Sentences)
    Set = set(words)
    # print(Set)
    len_w = len(Set)
    # print(len_w)
    del Set
    vocabulary_size = len_w+1
    data, count, dictionary, reverse_dictionary = build_dataset(words)
    # print(dictionary)
    print('Most common words (+UNK)', count[:5])

    batch_size = 1000
    embedding_size = 128
    context_window = 2
    context_size = 2*context_window

    # Number of negative samples
    num_sampled = 64

    graph = tf.Graph()
    with graph.as_default():
        train_inputs = tf.placeholder(tf.int32, shape =[batch_size, context_size])
        train_outputs = tf.placeholder(tf.int32, shape =[batch_size, 1])
        with tf.device('/cpu:0'):
            embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size],-1.0,1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)
            embed_context = tf.reduce_mean(embed,1)
            nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0/math.sqrt(embedding_size)))
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
            loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights,nce_biases, embed_context, train_outputs, num_sampled, vocabulary_size))
            optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = embeddings / norm
            init = tf.initialize_all_variables()

    num_steps = 10001

    with tf.Session(graph=graph) as session:

        init.run()
        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_outputs = generate_batch(data, batch_size, context_window)
            feed_dict = {train_inputs: batch_inputs, train_outputs: batch_outputs}
            _, loss_val = session.run([optimizer, loss], feed_dict = feed_dict)
            average_loss += loss_val
            if(step%2000==0):
                if step > 0:
                    average_loss /= 2000
                print("Avg Loss at Step ", step," =  ", average_loss)
        e =  embeddings.eval()
        save_pkl_file("embeddings_cbow.pkl",e)
        save_pkl_file("dictionary_cbow.pkl",dictionary)
        save_pkl_file("rev_dictionary_cbow.pkl",reverse_dictionary)
        return e, dictionary, reverse_dictionary

# generate_embeddings([['Sentences', 'asdfad','asdfasdfas', 'asdfasdfsadfsdf','asd2122']])
