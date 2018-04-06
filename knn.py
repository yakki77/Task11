import math
import operator
import csv

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
    max_list=[]
    min_list=[]
    for i in range(4):
        vector = []
        for j in range(len(testSet)):
            vector.append(testSet[j][i+2])
        for k in range(len(trainingSet)):
            vector.append(trainingSet[k][i+2])
        max_list.append(max_num_in_list(vector))
        min_list.append(min_num_in_list(vector))
    for i in range(len(testSet)):
        for j in range(4):
            testSet[i][j+2] = count(max_list[j],min_list[j],testSet[i][j+2])
    for i in range(len(trainingSet)):
        for j in range(4):
            trainingSet[i][j+2] = count(max_list[j],min_list[j],trainingSet[i][j+2])

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
        if index == 0 or index == 1:
            if vector1[index] == vector2[index]:
                tmp = 0
            else:
                tmp = 1
        else:
            tmp = vector1[index] - vector2[index]
        distance = distance + pow(tmp,2)
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
    for index in range(length):
    	label= neighbors[index][-1]
    	if label in frequency:
    		frequency[label] += 1
    	else:
    		frequency[label] = 1
    ans = sorted(frequency.items(), key=operator.itemgetter(1), reverse=True)
    return ans[0][0]

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

def getAccuracy(trainningInputPath,testInputPath,k,weightlist):
    trainingSet = []
    testSet = []
    loadData(trainningInputPath,testInputPath,trainingSet,testSet)
    length = len(testSet)
    sum = 0;
    correct = 0;
    for i in range(length):
        neighbors = findNeighborList(testSet[i],trainingSet,k,weightlist)
        label = getLabel(neighbors)
        sum += 1
        if label != testSet[i][-1]:
            print("not match: ",testSet[i],"predicted: "+label,"closest neighbor: ",neighbors[0])
            continue
        correct += 1
    print('Accuracy: ',correct/sum * 100,'%')
        

def main():
    k = 3;
    testInputPath = './self_test.csv'
    trainningInputPath = './trainCopy.csv'
    weightlist = []
    knn(trainningInputPath,testInputPath,k,weightlist)
    getAccuracy(trainningInputPath,testInputPath,k,weightlist)

main()