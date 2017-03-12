import os;
import re;
from collections import defaultdict;



def CustomerReview1(lines):
    Result = []
    for line in lines:
        match = re.search("##", line)
        if(match):
            data =  line.split('##')
            sentence = data[1].rstrip()
            features = data[0]
            featureListWithWeights = []
            Hash = {}
            Hash['Sentence'] = sentence
            Hash['PositiveFeatures'] = []
            Hash['NegativeFeatures'] = []
            if(len(features) > 0):
                for feature in features.rstrip().split(","):
                    negative = re.search('(.*)([[])([-])([0-9]+)(])', feature)
                    positive = re.search('(.*)([[])([+])([0-9]+)(])', feature)
                    if(positive):
                        Hash['PositiveFeatures'].append( ( positive.group(1),int(positive.group(4)) ) ) 
                    elif(negative):
                        Hash['NegativeFeatures'].append( ( negative.group(1),int(negative.group(4)) ) ) 
            Result.append(Hash)
    return Result

def parseFileForSentimentAnalysis(lines, Dataset):
    return Dataset(lines)

def returnDataSets():
    # f = open("data/EnglishSentimentDataSet/CustomerReview1/Readme.txt", "r") 
    # print f.read()
    filePaths = ['CustomerReview1/', 'CustomerReviewIJCAI/', 'ProductReview/']
    path = "data/EnglishSentimentDataSet/"
    for filepath in filePaths:
        directory = path+filepath
        files = os.listdir(directory)
        Datasets = []
        if 'Readme.txt' in files:
            files.remove('Readme.txt')
        for fileName in files:
            if fileName.endswith(".txt"):
                f = open(directory+fileName, "r")
                lines = f.readlines()
                DS = parseFileForSentimentAnalysis(lines, CustomerReview1)
                Datasets.append( DS )
        return Datasets
# print returnDataSets()