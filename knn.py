import math
import operator
import csv
import numpy as np

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
    testSet = np.matrix(testSet)
    va_list = [float(i[0,0]) for i in testSet[:,2]]
    va_list.sort()

    creadit_list = [float(i[0,0]) for i in testSet[:,3]]
    creadit_list.sort()

    salary_list = [float(i[0,0]) for i in testSet[:,4]]
    salary_list.sort()

    property_list = [float(i[0,0]) for i in testSet[:,5]]
    property_list.sort()

    for i in range(len(testSet[:,0])):
        person = []
        person.append(testSet[:,0][i])
        person.append(testSet[:,1][i])
        va = count(va_list[-1], va_list[0], testSet[i,2])
        credit = count(creadit_list[-1], creadit_list[0], testSet[i,3])
        salary = count(salary_list[-1], salary_list[0], testSet[i,4])
        property_ = count(property_list[-1], property_list[0], testSet[i,5])
        label = testSet[:,6][i]
        person.append(va)
        person.append(credit)
        person.append(salary)
        person.append(property_)
        person.append(label)
        person_list.append(person)
    return person_list


def count(max_num, min_num, num):
    num = float(num)
    return (num - min_num) / (max_num - min_num)

def euclidDistance(vector1,vector2):
    length = len(vector1)
    distance = 0
    for index in range(length-1):
        
        if index == 0 or index == 1:
            if vector1[index] == vector2[index]:
                tmp = 1
            else:
                tmp = 0
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
    return sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)[0][0]

def main():
    trainningSet = loadData('./trainProdSelection.csv')
    testSet = loadData('./testProdSelection.csv')
    k = 8
    predictions = []
    length = len(testSet)
    for i in range(length):
    	neighbors = findNeighborList(testSet[i],trainningSet,k)
    	label = getLabel(neighbors)
    	predictions.append(label)
    print(predictions)

main()