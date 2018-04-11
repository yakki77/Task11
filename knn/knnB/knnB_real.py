import math
import operator
import csv
import numpy as np

def loadData(trainingInputPath, testInputPath):
    trainingSet = []
    testSet = []
    with open(testInputPath, "r") as csvfile:
        test_data = csv.reader(csvfile, delimiter=' ', quotechar='|')
        index = -1
        for row in test_data:
            if index >= 0:
                features = row[0].split(',')
                testSet.append(features)
            index += 1
    with open(trainingInputPath, "r") as csvfile:
        test_data = csv.reader(csvfile, delimiter=' ', quotechar='|')
        index = -1
        for row in test_data:
            if index >= 0:
                features = row[0].split(',')
                trainingSet.append(features)
            index += 1
    return preprocess(trainingSet,testSet)

def preprocess(trainingSet,testSet):
    set = []
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
    set.append(trainingSet)
    set.append(testSet)
    return set


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

def euclidDistance(vector1,vector2,weightlist):
    length = len(vector1)
    distance = 0
    for index in range(length-1):
        if type(vector1[index]) == str:
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
    set = loadData(trainningInputPath,testInputPath)
    trainingSet = set[0]
    testSet = set[1]
    predictions = []
    classfier = []
    length = len(testSet)
    for i in range(length):
        neighbors = findNeighborList(testSet[i],trainingSet,k,weightlist)
        label = getLabel(neighbors)
        if label >= 20:
            classfier.append(1)
        else:
            classfier.append(0)
        predictions.append(label)
    print(predictions)
    print(classfier)
    return predictions

def mserror(real,pred):
    length = len(real)
    sum = 0
    for i in range(length):
        sum += pow(float(real[i]) - float(pred[i]),2)
    mse = sum / length
    return mse

def n_fold(trainingInputPath,n):
    datasets = {}
    total_set = []
    with open(trainingInputPath, "r") as csvfile:
        test_data = csv.reader(csvfile, delimiter=' ', quotechar='|')
        index = -1
        for row in test_data:
            if index >= 0:
                features = row[0].split(',')
                total_set.append(features)
            index += 1
    copy_total = total_set[:]
    each_count = (int)(len(total_set)/n)
    position = 0
    for i in range(n):
        set = []
        training_set=[]
        testing_set=[]
        total_set = copy_total[:]
        for j in range(each_count):
            if (position + j) < len(total_set)-1:
                testing_set.append(total_set[position+j])
                total_set.pop(position+j)
        position += each_count
        training_set = total_set
        datasets[i] = preprocess(total_set,testing_set)
    return datasets

def getMSE(trainingSet,testSet,k,weightlist):
    length = len(testSet)
    pred = []
    real = []
    for i in range(length):
        neighbors = findNeighborList(testSet[i],trainingSet,k,weightlist)
        label = getLabel(neighbors)
        pred.append(label)
        real.append(testSet[i][-1])
    mse = mserror(real,pred)
    return mse
 
def get_average_mse(datasets,k,n,weightlist):
    sum = 0
    for i in range(n):
        set = datasets[i]
        training_set = set[0]
        testing_set = set[1]
        acc = getMSE(training_set,testing_set,k,weightlist)
        sum += acc
    return sum/n

def evaluate(trainingInputPath,k,n,weightlist):
    datasets = n_fold(trainingInputPath,n)
    acc = get_average_mse(datasets,k,n,weightlist)
    print("mse: ",acc)

def main():
    k = 5
    n = 150
    testInputPath = './testProdIntro_real.csv'
    trainingInputPath = './trainProdIntro_real.csv'
    weightlist = [1,1,1.24,1,1,1,1.4,4]
    knn(trainingInputPath,testInputPath,k,weightlist)
    #evaluate(trainingInputPath,k,n,weightlist)


main()
