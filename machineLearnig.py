<<<<<<< HEAD
import csv
import random
import math
import operator


def numberfy(word, list):
    if word not in list:
        list.append(word)
        return len(list)
    else:
        return list.index(word)


def loadDataset(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'rt') as csvfile:
        crimes = []
        neighborhoods = []
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(1,len(dataset) - 1):
            for y in range(4,18):
                if y == 4:
                    print(dataset[x][y])
                    dataset[x][y] = numberfy(dataset[x][y],crimes)
                    print(dataset[x][y])
                    dataset[x][y] = float(dataset[x][y])
                elif y == 16:
                    dataset[x][y] = numberfy(dataset[x][y], neighborhoods)


            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])


def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def main():
    # prepare data
    trainingSet = []
    testSet = []
    split = 0.67
    loadDataset('crime.csv', split, trainingSet, testSet)
    print('Train set: ' + repr(len(trainingSet)))
    print('Test set: ' + repr(len(testSet)))
    # generate predictions
    predictions = []
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')


main()
=======
import pandas as pd

import matplotlib.pyplot as plt
from Tools.scripts.dutree import display
from scipy import stats
import numpy as np


names = ["INCIDENT_ID", "OFFENSE_ID", "OFFENSE_CODE", "OFFENSE_CODE_EXTENSION",
         "OFFENSE_TYPE_ID", "OFFENSE_CATEGORY_ID", "FIRST_OCCURRENCE_DATE",
         "LAST_OCCURRENCE_DATE", "REPORTED_DATE", "INCIDENT_ADDRESS",
         "GEO_X", "GEO_Y", "GEO_LON", "GEO_LAT", "DISTRICT_ID", "PRECINCT_ID",
         "NEIGHBORHOOD_ID", "IS_CRIME", "IS_TRAFFIC"]

#data = pd.read_csv("db/crime.csv",sep= ",",)
data=pd.read_csv('db/crime.csv', parse_dates=True)

data.info()

print(data)

temp=display(data.groupby([data.OFFENSE_CODE,data.OFFENSE_CODE_EXTENSION,data.OFFENSE_TYPE_ID]).size())
pd.set_option('display.max_rows',500)
print(temp)



>>>>>>> 6e97d703bf66401691c54c98744efb1aa33115bd
