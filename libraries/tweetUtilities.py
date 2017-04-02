import re,string
from nltk.corpus import stopwords
def strip_links(text):
    link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links         = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ', ')    
    return text

def strip_all_entities(text):
    entity_prefixes = ['@','#']
    for separator in  string.punctuation:
        if separator not in entity_prefixes :
            text = text.replace(separator,' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)

def tweetPreProcess(tweet, language='english'):
    tweet = strip_all_entities(tweet)
    tweet = strip_links(tweet)
    print tweet
    stop_words = stopwords.words(language)
    tweetTokens = [word for word in tweet.split() if word not in stop_words]
    print tweetTokens
    return ' '.join(tweetTokens)

def tweetTokenizer(tweet):
    tweet = tweetPreProcess(tweet)
    return tweet.split()  

