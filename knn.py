import math
import operator
import csv

def loadData(file_path):
    testSet = []
    person_list = []
    with open(file_path, "r") as csvfile:
        test_data = csv.reader(csvfile, delimiter=' ', quotechar='|')
        index = -1
        for row in test_data:
            if index >= 0:
                features = row[0].split(',')
                testSet.append(features)
            index += 1
    max_list=[]
    min_list=[]
    for i in range(4):
        vector = []
        for j in range(len(testSet)):
            vector.append(testSet[j][i+2])
        max_list.append(max_num_in_list(vector))
        min_list.append(min_num_in_list(vector))
    for i in range(len(testSet)):
        for j in range(4):
            testSet[i][j+2] = count(max_list[j],min_list[j],testSet[i][j+2])
    return testSet

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

def euclidDistance(vector1,vector2):
    length = len(vector1)
    distance = 0
    for index in range(length-1):
        if index == 0 or index == 1:
            if vector1[index] == vector2[index]:
                tmp = 0
            else:
                tmp = 1
        else:
            tmp = vector1[index] - vector2[index]
        distance = distance + pow(tmp,2)
    return math.sqrt(distance)

def findNeighborList(testVector,trainningSet,k):
    neighbors = []
    length = len(trainningSet)
    for index in range(length-1):
    	distance = euclidDistance(trainningSet[index],testVector)
    	neighbors.append((trainningSet[index],distance))
    neighbors.sort(key=operator.itemgetter(1))
    ans = []
    for neighbor in range(k):
    	ans.append(neighbors[k][0])
    return ans

def getLabel(neighbors):
    frequency={}
    length = len(neighbors)
    for index in range(length):
    	label= neighbors[index][-1]
    	if label in frequency:
    		frequency[label] += 1
    	else:
    		frequency[label] = 1
    return sorted(frequency.items(), key=operator.itemgetter(1), reverse=True)[0][0]

def knn(trainningInputPath,testInputPath,k):
    trainningSet = loadData(trainningInputPath)
    testSet = loadData(testInputPath)
    predictions = []
    length = len(testSet)
    for i in range(length):
    	neighbors = findNeighborList(testSet[i],trainningSet,k)
    	label = getLabel(neighbors)
    	predictions.append(label)
    print(predictions)

def getAccuracy(trainningInputPath,testInputPath,k):
    trainningSet = loadData(trainningInputPath)
    testSet = loadData(testInputPath)
    length = len(testSet)
    sum = 0;
    correct = 0;
    for i in range(length):
        neighbors = findNeighborList(testSet[i],trainningSet,k)
        label = getLabel(neighbors)
        if label == testSet[i][-1]:
            correct += 1
        sum += 1
    print('Accuracy: ',correct/sum * 100,'%')
        

def main():
    k = 1;
    testInputPath = './trainProdSelection.csv'
    trainningInputPath = './trainProdSelection.csv'
    knn(trainningInputPath,testInputPath,k)
    getAccuracy(trainningInputPath,testInputPath,k)
    
main()