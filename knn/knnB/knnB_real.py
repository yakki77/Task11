import math
import operator
import csv
import numpy as np

def loadData(trainningInputPath, testInputPath, trainingSet=[] , testSet=[]):
    with open(testInputPath, "r") as csvfile:
        test_data = csv.reader(csvfile, delimiter=' ', quotechar='|')
        index = -1
        for row in test_data:
            if index >= 0:
                features = row[0].split(',')
                testSet.append(features)
            index += 1
    with open(trainningInputPath, "r") as csvfile:
        test_data = csv.reader(csvfile, delimiter=' ', quotechar='|')
        index = -1
        for row in test_data:
            if index >= 0:
                features = row[0].split(',')
                trainingSet.append(features)
            index += 1
    max_list={}
    min_list={}
    num_column=[2,3,6,7]
    for i in num_column:
        vector = []
        for j in range(len(testSet)):
            vector.append(testSet[j][i])
        for k in range(len(trainingSet)):
            vector.append(trainingSet[k][i])
        max_list[i]=(max_num_in_list(vector))
        min_list[i]=(min_num_in_list(vector))
    for i in range(len(testSet)):
        for j in num_column:
            testSet[i][j] = count(max_list[j],min_list[j],testSet[i][j])
    for i in range(len(trainingSet)):
        for j in num_column:
            trainingSet[i][j] = count(max_list[j],min_list[j],trainingSet[i][j])

def max_num_in_list(list):
    max = float(list[0])
    for a in list:
        a = float(a)
        if a > max:
            max = a
    return max
def min_num_in_list(list):
    min = float(list[0])
    for a in list:
        a = float(a)
        if a < min:
            min = a
    return min

def count(max_num, min_num, num):
    num = float(num)
    max_num = float(max_num)
    min_num = float(min_num)
    return (num - min_num) / (max_num - min_num)

# next step: imlement distance calculation based on weightlist
def euclidDistance(vector1,vector2,weightlist):
    length = len(vector1)
    distance = 0
    for index in range(length-1):
        if index == 0 or index == 1 or index == 4 or index == 5:
            if vector1[index] == vector2[index]:
                tmp = 0
            else:
                tmp = 1
        else:
            tmp = vector1[index] - vector2[index]
        distance = distance + pow(tmp,2)*weightlist[index]
    return math.sqrt(distance)

def findNeighborList(testVector,trainningSet,k,weightlist):
    neighbors = []
    length = len(trainningSet)
    for index in range(length):
        distance = euclidDistance(trainningSet[index],testVector,weightlist)
        neighbors.append((trainningSet[index],distance))
    neighbors.sort(key=operator.itemgetter(1))
    ans = []
    for neighbor in range(k):
        ans.append(neighbors[neighbor][0])
    return ans

def getLabel(neighbors):
    frequency={}
    length = len(neighbors)
    sum = 0
    for index in range(length):
        score = neighbors[index][-1]
        sum += float(score)
    return sum/length

def knn(trainningInputPath,testInputPath,k,weightlist):
    trainingSet = []
    testSet = []
    loadData(trainningInputPath,testInputPath,trainingSet,testSet)
    predictions = []
    length = len(testSet)
    for i in range(length):
        neighbors = findNeighborList(testSet[i],trainingSet,k,weightlist)
        label = getLabel(neighbors)
        predictions.append(label)
    print(predictions)

def mserror(real,pred):
    length = len(real)
    sum = 0
    for i in range(length):
        sum += pow(float(real[i]) - float(pred[i]),2)
    mse = sum / length
    return mse
def getAccuracy(trainningInputPath,testInputPath,k,weightlist):
    trainingSet = []
    testSet = []
    loadData(trainningInputPath,testInputPath,trainingSet,testSet)
    length = len(testSet)
    pred = []
    real = []
    for i in range(length):
        neighbors = findNeighborList(testSet[i],trainingSet,k,weightlist)
        label = getLabel(neighbors)
        pred.append(label)
        real.append(testSet[i][-1])
    mse = mserror(real,pred)
    print('MSE: ',mse)
        

def main():
    k = 5;
    testInputPath = './trainProdIntro_real.csv'
    trainningInputPath = './trainProdIntro_real.csv'
    weightlist = [1,1,1,2,2,1,2,1]
    knn(trainningInputPath,testInputPath,k,weightlist)
    getAccuracy(trainningInputPath,testInputPath,k,weightlist)

main()