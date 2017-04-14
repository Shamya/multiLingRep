# coding: utf-8

# In[2]:


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
import ujson
import gensim
from libraries.acs import acs,acs_map

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

# Path to the Language Wiki Corpii

ENG_DIR = os.path.join(os.getcwd(),"data\\English_Wiki")
ESP_DIR = os.path.join(os.getcwd(),"wikiextractor\\Spanish_Wiki")
FRE_DIR = os.path.join(os.getcwd(),"wikiextractor\\French_Wiki")
ITA_DIR = os.path.join(os.getcwd(),"wikiextractor\\Italian_Wiki")
# File list in each directory

ENG_filenames=os.listdir(ENG_DIR)
ESP_filenames=os.listdir(ESP_DIR)
FRE_filenames=os.listdir(FRE_DIR)
ITA_filenames=os.listdir(ITA_DIR)


# Global list variable for English sentences after processing
ENG_sentences = []
# Global list variable for Spanish sentences after processing
ESP_sentences = []
# Global list variable for French sentences after processing
FRE_sentences = []
# Global list variable for Italian sentences after processing
ITA_sentences = []

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
    
    
    
def spanishcorpus(mode):
    # Global list variable for Spanish sentences after processing
    ESP_sentences = []
    if mode=="Load":
        # Reading data back
       print "Loading Spanish"
       with open('data/espdata.json', 'r') as f:
             ESP_sentences = ujson.load(f)
       f.close()
       print "Spanish Loaded"   
  
    print "Done with Spanish"
    if mode=="Save":
        print "Going through Spanish Wiki"
        for subdir, dirs, files in os.walk(ESP_DIR):
            # print subdir, dirs, files, files[::5]
            for file in files[::5]:
                with codecs.open(os.path.join(subdir,file),'rb',encoding='utf-8') as espdoc:
                    espcontent =espdoc.read()
                    ESP_sentences += preprocess(espcontent)
        # Random shuffle of the sentences
        shuffle(ESP_sentences)
        # Saving to file
        print "Pickling Spanish"
        
        # Writing JSON data
        with open('data/espdata.json', 'w') as f:
             ujson.dump(ESP_sentences, f)
        f.close()
    return ESP_sentences    

def frenchcorpus(mode):
    if mode=="Load":
        # Reading data back
        with open('fredata.json', 'r') as f:
             FRE_sentences = ujson.load(f)
        f.close()
    
    print "Done with French"
    if mode=="Save":
        print "Going through French Wiki"
        for subdir, dirs, files in os.walk(FRE_DIR):
            for file in files:
                with codecs.open(os.path.join(subdir,file),'rb',encoding='utf-8') as fredoc:
                    frecontent = fredoc.read()
                    print "French"
                    FRE_sentences += preprocess(frecontent)
        # Random shuffle of the sentences
        shuffle(FRE_sentences)
        # Saving to file
        print "Pickling French"
        
        # Writing JSON data
        with open('fredata.json', 'w') as f:
             ujson.dump(FRE_sentences, f)
        f.close()
        
def italiancorpus(mode):
    if mode=="Load":
        # Reading data back
        with open('itadata.json', 'r') as f:
             ITA_sentences = ujson.load(f)
        f.close()
    
    print "Done with Italian"
    if mode=="Save":
        print "Going through Italian Wiki"
        for subdir, dirs, files in os.walk(ITA_DIR):
            for file in files:
                with codecs.open(os.path.join(subdir,file),'rb',encoding='utf-8') as itadoc:
                    itacontent =itadoc.read()
                    print "Italian"
                    ITA_sentences += preprocess(itacontent)
        # Random shuffle of the sentences
        shuffle(ITA_sentences)
        # Saving to file
        print "Pickling Italian"
        
        # Writing JSON data
        with open('itadata.json', 'w') as f:
             ujson.dump(ITA_sentences, f)
        f.close()   
   

def englishcorpus(mode):
    # Global list variable for English sentences after processing
    ENG_sentences = []
    
    if mode=="Load":
        # Reading data back
        print "Loading English"
        with open('data/engdata.json', 'r') as f:
             ENG_sentences = ujson.load(f)
        f.close()
        print "Done with English"
               
    if mode=="Save":
        print "Going through English Wiki"            
            
        for subdir, dirs, files in os.walk(ENG_DIR):
       
           for file in files[::5]:
        
                with codecs.open(os.path.join(subdir,file),'rb',encoding='utf-8') as engdoc:
                    engcontent =engdoc.read()
                    ENG_sentences += preprocess(engcontent)
                
        # Saving to file
        print "Pickling English"
        
        # Writing JSON data
        with open('data/engdata.json', 'w') as f:
             ujson.dump(ENG_sentences, f)
        f.close()
    print "Done saving"
    return ENG_sentences    
        

# In[141]:

"""
#Implement Code Switching here 

def ACS_sentences(ENG_sentences):
    for line in ENG_sentences:
        res=[]
        for word in line:
            x = acs(word)
            res.append(x)
        ENG_sentences[ENG_sentences.index(line)]=res 
    return ENG_sentences

#ENG_sentences = ACS_sentences(ENG_sentences)
"""
def transform(sentences, language):
    for sentence in sentences:
        for ind in xrange(len(sentence)):
            sentence[ind] = acs_map(sentence[ind], language)
    return sentences


def multilingualcorpus(mode,ENG_sentences,ESP_sentences):
    multilingual_data=[]
    if mode=="Save":
        
        # Assimilated Corpus of Multilingual texts        
        multilingual_data= transform(ENG_sentences,'spanish') + transform(ESP_sentences, 'spanish')
        # Random shuffle of the sentences
        shuffle(multilingual_data)
        # Saving to file
        print "Pickling Multi-Corpus"
        
        # Writing JSON data
        with open('multidata.json', 'w') as f:
             ujson.dump(multilingual_data, f)
        f.close()
    print "Corpus saved"
        
    if mode=="Load":
        # Reading data back
        print "Load Corpus"
        with open('multidata.json', 'r') as f:
             multilingual_data = ujson.load(f)
        f.close()
        print "Done with Multi-Lingual Corpus"
    return multilingual_data


def embeddings(ENG_sentences,ESP_sentences,multilingual_data):
    """ 
    Using the word2vec implementation   
    - Initialize a model
    Parameters:
    - (sg=0), CBOW is used. Otherwise (sg=1), skip-gram is employed
    - size is the dimensionality of the feature vectors.
    - window is the maximum distance between the current and predicted word within a sentence.
    - min_count => ignore all words with total frequency lower than this.
    
    """
    # Persist a model to disk
    
    fname0=os.path.join(os.getcwd(),'word_vectors_eng.txt')
    fname1=os.path.join(os.getcwd(),'word_vectors_esp.txt')
    fname2=os.path.join(os.getcwd(),'word_vectors_fre.txt')
    fname3=os.path.join(os.getcwd(),'word_vectors_ita.txt')
    fnamex=os.path.join(os.getcwd(),'word_vectors_mul.txt')
    
    # Word2Vec
    print "Saving English model"
    model_eng = models.Word2Vec(ENG_sentences, size=300, window=5, min_count=5, workers=4)
    model_eng.save(fname0)
    print "Saving Spanish model"
    model_esp = models.Word2Vec(ESP_sentences, size=300, window=5, min_count=5, workers=4)
    model_esp.save(fname1)
    print "Saving French model"
    model_fre = models.Word2Vec(FRE_sentences, size=300, window=5, min_count=5, workers=4)
    model_fre.save(fname2)
    print "Saving Italian model"
    model_ita = models.Word2Vec(ITA_sentences, size=300, window=5, min_count=5, workers=4)
    model_ita.save(fname3)
    print "Saving multi-lingual model"
    model_mul = models.Word2Vec(multilingual_data, size=300, window=5, min_count=5, workers=4)
    model_mul.save(fnamex)

def evaluate_embeddings(model):
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
    accudict= model.accuracy(os.path.join(os.getcwd(),'questions-words.txt'))
    for i in range(len(accudict)):
        print "For category ", accudict[i]['section'], "Accuracy is ", 100*float(len(accudict[i]['correct']))/(len(accudict[i]['incorrect'])+len(accudict[i]['correct']))



def corpus_stats():
    
    # Filenames of the saved models
    fname0 = os.path.join(os.getcwd(),'word_vectors_eng.txt')
    fname1 = os.path.join(os.getcwd(),'word_vectors_esp.txt')
    fname2 = os.path.join(os.getcwd(),'word_vectors_fre.txt')
    fname3 = os.path.join(os.getcwd(),'word_vectors_ita.txt')
    fnamex = os.path.join(os.getcwd(),'word_vectors_mul.txt')
    # Loading the models
    model_eng = gensim.models.Word2Vec.load(fname0)
    model_esp = gensim.models.Word2Vec.load(fname1)
    model_fre = gensim.models.Word2Vec.load(fname2)
    model_ita = gensim.models.Word2Vec.load(fname3)
    model_mul = gensim.models.Word2Vec.load(fnamex)
    
    vocab_eng = list(model_eng.wv.vocab.keys())
    vocab_esp = list(model_esp.wv.vocab.keys())
    vocab_fre = list(model_fre.wv.vocab.keys())
    vocab_ita = list(model_ita.wv.vocab.keys())
    vocab_mul = list(model_mul.wv.vocab.keys())
    
    print "\n Vocabulary length of English Corpus : ", len(vocab_eng)
    print "\n Vocabulary length of Spanish Corpus : ", len(vocab_esp)
    print "\n Vocabulary length of French Corpus : ", len(vocab_fre)
    print "\n Vocabulary length of Italian Corpus : ", len(vocab_ita)
    print "\n Vocabulary length of Multi-Lingual Corpus : ", len(vocab_mul)
    
    print "Evaluating English embeddings "
    evaluate_embeddings(model_eng)
    print "\n\n"
    print "Evaluating Spanish embeddings "
    evaluate_embeddings(model_esp)
    print "\n\n"
    print "Evaluating Multi-lingual embeddings "
    evaluate_embeddings(model_mul)
    

        
"""
Uncomment the corresponding lines to run the respective code
"""        
#ESP_sentences=spanishcorpus("Save")
#ESP_sentences=spanishcorpus("Load")

#FRE_sentences=frenchcorpus("Save")
#FRE_sentences=frenchcorpus("Load")

#ITA_sentences=italiancorpus("Save")
#ITA_sentences=italiancorpus("Load")

#ENG_sentences = englishcorpus("Save")
#ENG_sentences = englishcorpus("Load")


# Number of languages
L=4
corpus= np.zeros((L,1))
corpus[0] = ENG_sentences
corpus[1] = ESP_sentences
corpus[2] = FRE_sentences
corpus[3] = ITA_sentences


#multilingual_data = multilingualcorpus("Save",ENG_sentences, ESP_sentences)
#multilingual_data = multilingualcorpus("Load")



#embeddings(ENG_sentences,ESP_sentences,multilingual_data)
#corpus_stats()
