import math
import operator
import csv
import sys

#loading data from file path
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


#normalize dataset
def preprocess(trainingSet,testSet):
    set = []
    max_list={}
    min_list={}
    num_column=[2,3,4,5]
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

#find the max value in a list
def max_num_in_list(list):
    max = float(list[0])
    for a in list:
        a = float(a)
        if a > max:
            max = a
    return max

#find the min value in a list
def min_num_in_list(list):
    min = float(list[0])
    for a in list:
        a = float(a)
        if a < min:
            min = a
    return min

#return the scaled value of an attribute
def count(max_num, min_num, num):
    num = float(num)
    max_num = float(max_num)
    min_num = float(min_num)
    return (num - min_num) / (max_num - min_num)

#caculate euclid Distance
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

#find the neighbor list based on a test vector and training set
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

#get the label based on current neighbors list
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

# main logic of knn algorithm
def knn(k,trainingInputPath,testInputPath):
    weightlist = [0.1,0.0,0.1,4.1,0.4,1.8]
    set = loadData(trainingInputPath,testInputPath)
    trainingSet = set[0]
    testSet = set[1]
    predictions = []
    length = len(testSet)
    for i in range(length):
        neighbors = findNeighborList(testSet[i],trainingSet,k,weightlist)
        label = getLabel(neighbors)
        predictions.append(label)
    print(predictions)

# get the accuracy for test set
def getAccuracy(trainingSet,testSet,k,weightlist):
    length = len(testSet)
    sum = 0;
    correct = 0;
    for i in range(length):
        neighbors = findNeighborList(testSet[i],trainingSet,k,weightlist)
        label = getLabel(neighbors)
        sum += 1
        if label != testSet[i][-1]:
            continue
        correct += 1
    return correct/sum

# split the data into n folds
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

#get average accuracy for given weightlist
def get_average_accuracy(datasets,k,n,weightlist):
    sum = 0
    for i in range(n):
        set = datasets[i]
        training_set = set[0]
        testing_set = set[1]
        acc = getAccuracy(training_set,testing_set,k,weightlist)
        sum += acc
    return sum/n

# test method 
def evaluate(k,trainingInputPath):
    weightlist = [0.1,0.0,0.1,4.1,0.4,1.8]
    n = 180
    datasets = n_fold(trainingInputPath,n)
    acc = get_average_accuracy(datasets,k,n,weightlist)
    print(acc * 100,"%")


def main(action,trainingInputPath,testInputPath):
    k = 3
    if action == 'prediction':
        knn(k,trainingInputPath,testInputPath)
    elif action =='validation':
        evaluate(k,trainingInputPath)
    else:
        print('invalid argument')

if __name__== "__main__":
    action = sys.argv[1]
    trainingInputPath = sys.argv[2]
    testInputPath = sys.argv[3]
    main(action,trainingInputPath,testInputPath)