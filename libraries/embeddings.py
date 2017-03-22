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
from libraries.acs import acs
import time
import pickle

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

# print os.path.join(os.getcwd(),"/wikiDS/English/AA/AA")
ENG_DIR = "/Users/danielsampetethiyagu/github/OracleMultiLing/wikiDS/English/AA/AA"
ESP_DIR = "/Users/danielsampetethiyagu/github/OracleMultiLing/wikiDS/Spanish/AA/AA"

# File list in each directory

ENG_filenames=os.listdir(ENG_DIR)
ESP_filenames=os.listdir(ESP_DIR)


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


def ACS(multilingual_data):
    for line_number in xrange(len(multilingual_data)):
        line = multilingual_data[line_number]
        res = []
        for word in line:
            res.append(acs(word))
        multilingual_data[line_number] = res
    return multilingual_data

def gather_from_dir(Directory):
    Sentences = []
    for f in os.walk(Directory):
        path, x, file_names = f
        for file_name in file_names:
            print(str(path)+"/"+str(file_name))
            with codecs.open(path+"/"+file_name,'rb',encoding='utf-8') as doc:
                content = doc.read()
                Sentences += preprocess(content)
    return Sentences


def gather_data():
    # Global list variable for English sentences after processing
    ENG_sentences = []
    # Global list variable for Spanish sentences after processing
    ESP_sentences = []
    ESP_sentences = gather_from_dir(ESP_DIR)
    ENG_sentences = gather_from_dir(ENG_DIR)
        
    # Assimilated Corpus of Multilingual texts        
    multilingual_data=[]        
    multilingual_data= ENG_sentences + ESP_sentences
    # Random shuffle of the sentences
    shuffle(multilingual_data)
    # multilingual_data = ACS(multilingual_data)
    return multilingual_data



def get_model(multilingual_data):

    # Persist a model to disk
    fname= "wikiDS/word2Vec.mdl"
    vocabfname = "wikiDS/vocab.pkl"
    """ 
    Using the word2vec implementation   
    - Initialize a model
    Parameters:
    - (sg=0), CBOW is used. Otherwise (sg=1), skip-gram is employed
    - size is the dimensionality of the feature vectors.
    - window is the maximum distance between the current and predicted word within a sentence.
    - min_count => ignore all words with total frequency lower than this.

    """

    model = models.Word2Vec(multilingual_data, size=128, window=5, min_count=5, workers=4)
    model.save(fname)

    vocab = list(model.vocab.keys())
    vocabfile = codecs.open(vocabfname, "w", "utf-8")
    pickle.dump( vocab, vocabfile )
    vocab_len = len(vocab)
    print("Vocab length is ",vocab_len);
    test_model(model)
    return model


def test_model(model):
    accudict= model.accuracy(os.path.join(os.getcwd(),'questions-words.txt'))

    for i in range(len(accudict)):
        if(len(accudict[i]['incorrect'])+len(accudict[i]['correct']) >0):
            print "For category ", accudict[i]['section'], "Accuracy is ", 100*float(len(accudict[i]['correct']))/(len(accudict[i]['incorrect'])+len(accudict[i]['correct']))


def process_and_retrieve_model():
    multilingual_data = gather_data()
    return get_model(multilingual_data)

