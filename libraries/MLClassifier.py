from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def RandForest():
  return RandomForestClassifier(n_estimators = 100)

def LR():
  return LogisticRegression(C=1e5)

def classifierModel(classifierFunction):
  return classifierFunction()