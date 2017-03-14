from random import shuffle
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
import pandas as pd;
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from MLClassifier import classifierModel
# train, test = train_test_split( data, train_size = 0.8, random_state = 44 )


def splitDataSets(DataSets, train=0.6, validationSize=0.1, test=0.3):
  TestSet = []
  ValidSet = []
  TrainSet = []
  for iterator in range(len(DataSets)):
    Dataset = DataSets[iterator]
    shuffle(Dataset)
    cnt = len(Dataset)
    trainlen = int(cnt*train)
    trainset = Dataset[:trainlen]
    validlen = trainlen+int(cnt*validationSize)
    validset = Dataset[trainlen:validlen]
    testset = Dataset[validlen:]
    TrainSet.append(trainset)
    ValidSet.append(validset)
    TestSet.append(testset)
  return TrainSet, ValidSet, TestSet



def prepareDataSetForSKlearn(TrainSet, ValidSet, TestSet, combineTrainValid=True):
  vec = DictVectorizer()
  TrainX = []
  TestX = []
  ValidX = []

  # Train =  list(itertools.chain.from_iterable(TrainSet))
  
  if(combineTrainValid):
    for i in range(len(ValidSet)):
      TrainSet.append(ValidSet[i])
  Train =  list(itertools.chain.from_iterable(TrainSet))
  train = pd.DataFrame.from_records(Train)


  Test =  list(itertools.chain.from_iterable(TestSet))
  test = pd.DataFrame.from_records(Test)

  vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 

  # fit_transform() does two functions: First, it fits the model
  # and learns the vocabulary; second, it transforms our training data
  # into feature vectors. The input to fit_transform should be a list of 
  # strings.
  train_data_features = vectorizer.fit_transform(train['Sentence'])

  # Numpy arrays are easy to work with, so convert the result to an 
  # array

  # classifier = classifierModel(RandForest, train_data_features, train["ClassifiedOutput"], )

  # Initialize a Random Forest classifier with 100 trees
  classifier = RandomForestClassifier(n_estimators = 100) 

  # Fit the forest to the training set, using the bag of words as 
  # features and the sentiment labels as the response variable
  #
  # This may take a few minutes to run
  classifier = classifier.fit( train_data_features, train["ClassifiedOutput"] )

  train_data_features = train_data_features.toarray()

  # Create an empty list and append the clean reviews one by one
  num_reviews = len(test["Sentence"])

  # Get a bag of words for the test set, and convert to a numpy array
  test_data_features = vectorizer.transform(test["Sentence"])
  test_data_features = test_data_features.toarray()

  # Use the random forest to make sentiment label predictions
  result = classifier.predict(test_data_features)
  print accuracy_score(test["ClassifiedOutput"], result)

  answers = list(test["ClassifiedOutput"])
  print accuracy_score(answers, result)
  # print result == answers
  # print "Accuracy of the Dataset is " + str()
  # Copy the results to a pandas dataframe with an "id" column and
  # a "sentiment" column
  output = pd.DataFrame( data={"Sentence":test["Sentence"], "sentiment":result, "ActualOutput": test["ClassifiedOutput"]} )

  # Use pandas to write the comma-separated output file
  output.to_csv( "data/Bag_of_Words_model.csv", sep='\t', encoding='utf-8')
  vocab = vectorizer.get_feature_names()
  return train, test

# tr, te = prepareDataSetForSKlearn(Train,Validation,Test)
