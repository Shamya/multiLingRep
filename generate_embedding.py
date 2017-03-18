
# coding: utf-8

# In[26]:


import os
import numpy as np
import nltk, re, pprint
import django
import codecs
from gensim import models
from django.utils.encoding import smart_str
from bs4 import BeautifulSoup
from nltk import tokenize
from random import shuffle



# In[ ]:

from gensim.parsing import PorterStemmer
global_stemmer = PorterStemmer()
 
class StemmingHelper(object):
    """
    Class to aid the stemming process - from word to stemmed form,
    and vice versa.
    The 'original' form of a stemmed word will be returned as the
    form in which its been used the most number of times in the text.
    """
 
    #This reverse lookup will remember the original forms of the stemmed
    #words
    word_lookup = {}
 
    @classmethod
    def stem(cls, word):
        """
        Stems a word and updates the reverse lookup.
        """
 
        #Stem the word
        stemmed = global_stemmer.stem(word)
 
        #Update the word lookup
        if stemmed not in cls.word_lookup:
            cls.word_lookup[stemmed] = {}
        cls.word_lookup[stemmed][word] = (
            cls.word_lookup[stemmed].get(word, 0) + 1)
 
        return stemmed
 
    @classmethod
    def original_form(cls, word):
        """
        Returns original form of a word given the stemmed version,
        as stored in the word lookup.
        """
 
        if word in cls.word_lookup:
            return max(cls.word_lookup[word].keys(),
                       key=lambda x: cls.word_lookup[word][x])
        else:
            return word


# In[33]:

# Path to the English and Spanish Wiki Corpii

ENG_DIR = os.path.join(os.getcwd(),"Corpus\\English\\English_Wiki\\AA")
ESP_DIR = os.path.join(os.getcwd(),"Corpus\\Spanish\\Spanish_Wiki\\AA")

# File list in each directory

ENG_filenames=os.listdir(ENG_DIR)
ESP_filenames=os.listdir(ENG_DIR)

# Global list variable for English sentences after processing
ENG_sentences = []
# Global list variable for Spanish sentences after processing
ESP_sentences = []


def preprocess(content):
    """
    Function that preprocesses the data 
    
    (1) Parses through html/xml content and removes it
    (2) Sets words to lowercase
    (3) Sentence Tokenization
    (4) Word Tokenization and Lemmatization
    
    Parameters: Content read from the file
    Returns : List of sentences 
    """
    soup = BeautifulSoup(content, 'html.parser')
    data = soup.get_text().lower()
    sentences=tokenize.sent_tokenize(data.lower())
    for i in range(len(sentences)):
        sentences[i]=sentences[i].split()
    return sentences

for file in ESP_filenames:
    with codecs.open(os.path.join(ESP_DIR,file),'rb',encoding='utf-8') as espdoc:
        espcontent =espdoc.read()
        ESP_sentences += preprocess(espcontent)

for file in ENG_filenames:
    with codecs.open(os.path.join(ENG_DIR,file),'rb',encoding='utf-8') as engdoc:
        engcontent =engdoc.read()
        ENG_sentences += preprocess(engcontent)
        
        
# Assimilated Corpus of Multilingual texts        
multilingual_data=[]        
multilingual_data= ENG_sentences + ESP_sentences

# Random shuffle of the sentences
shuffle(multilingual_data)


# In[141]:

"""
Implement Code Switching here 

"""


# In[88]:

for file in ENG_filenames[1:2]:
    with codecs.open(os.path.join(ENG_DIR,file),'rb',encoding='utf-8') as engdoc:
        engcontent = engdoc.read()
        
preprocess(engcontent)


# In[49]:

# Persist a model to disk
fname= "E:\\Learning\\Nlp\\Project stuff\\word_vectors.txt"

""" 
Using the word2vec implementation   
- Initialize a model
Parameters:
- (sg=0), CBOW is used. Otherwise (sg=1), skip-gram is employed
- size is the dimensionality of the feature vectors.
- window is the maximum distance between the current and predicted word within a sentence.
- min_count => ignore all words with total frequency lower than this.

"""

model = models.Word2Vec(multilingual_data, size=300, window=5, min_count=5, workers=4)
model.save(fname)


# In[36]:

vocab = list(model.vocab.keys())
vocab_len = len(vocab)
vocab_len


# In[55]:

st= vocab[448]
st


# In[56]:

model[st]


# In[ ]:

"""
Evaluating the Embeddings

-Performing NLP word tasks with the model
-Probability of a text under the model
-Correlation with human opinion on word similarity and on analogies
-Question-words: Google have released their testing set of about 20,000 syntactic and semantic test examples, 
following the “A is to B as C is to D” task
- Doesn't match
- Most similar


"""


# In[60]:

accudict= model.accuracy(os.path.join(os.getcwd(),'questions-words.txt'))


# In[81]:

for i in range(len(accudict)):
    print "For category ", accudict[i]['section'], "Accuracy is ", 100*float(len(accudict[i]['correct']))/(len(accudict[i]['incorrect'])+len(accudict[i]['correct']))



# In[83]:

model.most_similar(positive=['woman', 'king'], negative=['man'])


# In[85]:

model.similarity('woman', 'man')

