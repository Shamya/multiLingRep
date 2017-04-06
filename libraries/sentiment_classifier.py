from xml.dom.minidom import parse as pr
import pandas as pd
from libraries.tweetUtilities import tweetTokenizer, tweetPreProcess
from libraries.dataSetUtility import classify
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


def sentimentAnalysisFrenchDataset():
  train_data = sentimentData(TRAIN_FILES)
  print len(train_data)
  test_data = sentimentData(TEST_FILES)
  print len(test_data)

  train_df = pd.DataFrame(train_data, columns = ['Sentence' , 'ClassifiedOutput'])
  test_df = pd.DataFrame(test_data, columns = ['Sentence' , 'ClassifiedOutput'])

  classify(train_df, test_df)