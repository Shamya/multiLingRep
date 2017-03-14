import os;
import re;
from collections import defaultdict;
import io

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
                posCnt = 0;
                negCnt = 0;
                for feature in features.rstrip().split(","):
                    negative = re.search('(.*)([[])([-])([0-9]+)(])', feature)
                    positive = re.search('(.*)([[])([+])([0-9]+)(])', feature)
                    if(positive):
                        Hash['PositiveFeatures'].append( ( positive.group(1),int(positive.group(4)) ) ) 
                        posCnt+=int(positive.group(4))
                    elif(negative):
                        Hash['NegativeFeatures'].append( ( negative.group(1),int(negative.group(4)) ) )
                        negCnt+=int(negative.group(4))
                if(posCnt<=negCnt):
                    Hash['ClassifiedOutput'] = -1;
                else:
                    Hash['ClassifiedOutput'] = 1;
            if 'ClassifiedOutput' not in Hash:
                Hash['ClassifiedOutput'] = -1
            Result.append(Hash)
    return Result

def parseFileForSentimentAnalysis(lines, Dataset):
    return Dataset(lines)

def returnDataSets():
    # f = open("data/EnglishSentimentDataSet/CustomerReview1/Readme.txt", "r") 
    # print f.read()
    filePaths = ['CustomerReview1/', 'CustomerReviewIJCAI/', 'ProductReview/']
    path = "data/EnglishSentimentDataSet/"
    Datasets = []
    for filepath in filePaths:
        print filepath
        directory = path+filepath
        files = os.listdir(directory)
        if 'Readme.txt' in files:
            files.remove('Readme.txt')
        for fileName in files:
            print file
            if fileName.endswith(".txt"):
                print fileName
                f = io.open(directory+fileName, encoding="ISO-8859-1")
                lines = f.readlines()
                DS = parseFileForSentimentAnalysis(lines, CustomerReview1)
                Datasets.append( DS )
    return Datasets
# print returnDataSets()