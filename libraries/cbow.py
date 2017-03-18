
# coding: utf-8

# In[16]:

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


# In[7]:

Sentences = ['This is a good world', 'This is a really bad world']
def getWords(Sentences):
    words = []
    for sentence in Sentences:
        for word in sentence.split():
            words.append(word)
    return words

words = getWords(Sentences)


# In[9]:

print(words)


# In[4]:

vocabulary_size = 50000
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


# In[10]:

data, count, dictionary, reverse_dictionary = build_dataset(words)
print(dictionary)


# In[12]:

print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])


# In[25]:

batch_size = 500
embedding_size = 200
vocabulary_size = 2000
generations = 50000
model_learning_rate = 0.001
num_sampled = int(batch_size/2)
window_size = 3
# Add checkpoints to training
save_embeddings_every = 5000
print_valid_every = 5000
print_loss_every = 100
# Declare stop words
stops = stopwords.words('english')
# We pick some test words. We are expecting synonyms to appear
valid_words = ['love', 'hate', 'happy', 'sad', 'man', 'woman']
data_index = 0


# In[31]:

def generate_batch(batch_size, context_window):
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


# In[39]:

batch, labels = generate_batch(batch_size=8, context_window=2)


# In[42]:

for i in range(8):
    print(batch[i, 0], reverse_dictionary[batch[i, 0]], batch[i, 1], reverse_dictionary[batch[i, 1]],batch[i, 2], reverse_dictionary[batch[i, 2]], batch[i, 3], reverse_dictionary[batch[i, 3]], '->', labels[i, 0], reverse_dictionary[labels[i, 0]])


# In[43]:

batch_size = 128
embedding_size = 128
context_window = 2
context_size = 2*context_window

# Number of negative samples
num_sampled = 64


# In[50]:

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

        


# In[61]:

num_steps = 10001

with tf.Session(graph=graph) as session:

    init.run()
    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_outputs = generate_batch(batch_size, context_window)
        feed_dict = {train_inputs: batch_inputs, train_outputs: batch_outputs}
        _, loss_val = session.run([optimizer, loss], feed_dict = feed_dict)
        average_loss += loss_val
        if(step%2000==0):
            if step > 0:
                average_loss /= 2000
            print("Avg Loss at Step ", step," =  ", average_loss)
    e =  embeddings.eval()
        


# In[62]:

e[0][:]

