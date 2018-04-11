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

#get the label based on current neighbors list for binary test
def getLabelBinary(neighbors):
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

#get the label based on current neighbors list for real test
def getLabelReal(neighbors):
    frequency={}
    length = len(neighbors)
    sum = 0
    for index in range(length):
        score = neighbors[index][-1]
        sum += float(score)
    return sum/length

# main logic of knn algorithm for binary test
def knnBinary(trainningInputPath,testInputPath,k,weightlist):
    set = loadData(trainningInputPath,testInputPath)
    trainingSet = set[0]
    testSet = set[1]
    predictions = []
    length = len(testSet)
    for i in range(length):
        neighbors = findNeighborList(testSet[i],trainingSet,k,weightlist)
        label = getLabelBinary(neighbors)
        predictions.append(label)
    print(predictions)

# main logic of knn algorithm for real test
def knnReal(trainningInputPath,testInputPath,k,weightlist):
    set = loadData(trainningInputPath,testInputPath)
    trainingSet = set[0]
    testSet = set[1]
    predictions = []
    length = len(testSet)
    for i in range(length):
        neighbors = findNeighborList(testSet[i],trainingSet,k,weightlist)
        label = getLabelReal(neighbors)
        predictions.append(label)
    print(predictions)

# get the accuracy for test set
def getAccuracy(trainingSet,testSet,k,weightlist):
    length = len(testSet)
    sum = 0;
    correct = 0;
    for i in range(length):
        neighbors = findNeighborList(testSet[i],trainingSet,k,weightlist)
        label = getLabelBinary(neighbors)
        sum += 1
        if label != testSet[i][-1]:
            continue
        correct += 1
    return correct/sum

# calculate mse 
def mserror(real,pred):
    length = len(real)
    sum = 0
    for i in range(length):
        sum += pow(float(real[i]) - float(pred[i]),2)
    mse = sum / length
    return mse

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

#calculate mse for a given test set and weightlist
def getMSE(trainingSet,testSet,k,weightlist):
    length = len(testSet)
    pred = []
    real = []
    for i in range(length):
        neighbors = findNeighborList(testSet[i],trainingSet,k,weightlist)
        label = getLabelReal(neighbors)
        pred.append(label)
        real.append(testSet[i][-1])
    mse = mserror(real,pred)
    return mse
 
# get the average mse for given weightlist
def get_average_mse(datasets,k,n,weightlist):
    sum = 0
    for i in range(n):
        set = datasets[i]
        training_set = set[0]
        testing_set = set[1]
        acc = getMSE(training_set,testing_set,k,weightlist)
        sum += acc
    return sum/n

# get the average accuracy for given weightlist
def get_average_accuracy(datasets,k,n,weightlist):
    sum = 0
    for i in range(n):
        set = datasets[i]
        training_set = set[0]
        testing_set = set[1]
        acc = getAccuracy(training_set,testing_set,k,weightlist)
        sum += acc
    return sum/n

# test method for binary data
def evaluateBinary(trainingInputPath,k,n,weightlist):
    datasets = n_fold(trainingInputPath,n)
    acc = get_average_accuracy(datasets,k,n,weightlist)
    print(acc * 100,"%")

# test method for real data
def evaluateReal(trainingInputPath,k,n,weightlist):
    datasets = n_fold(trainingInputPath,n)
    acc = get_average_mse(datasets,k,n,weightlist)
    print("mse: ",acc)
 
# test method
def evaluate(k,type,trainingInputPath,testInputPath):
    n = 150
    testInputPathBinary = './testProdIntro_binary.csv'
    trainingInputPathBinary = './trainProdIntro_binary.csv'
    weightlistBinary = [0.02446, 0.1583, 0.01, 0.105, 0.09, 0.068, 0.25, 0.29]
    testInputPathReal = './testProdIntro_real.csv'
    trainingInputPathReal = './trainProdIntro_real.csv'
    weightlistReal = [1,1,1.24,1,1,1,1.4,4]
    if type == 'real':
        evaluateReal(trainingInputPathReal,k,n,weightlistReal)
    elif type == 'binary':
        evaluateBinary(trainingInputPathBinary,k,n,weightlistBinary)
    else:
        print("invalid type!")

#main logic of knn
def knn(k,t,trainingInputPath,testInputPath):
    weightlistBinary = [0.02446, 0.1583, 0.01, 0.105, 0.09, 0.068, 0.25, 0.29]
    weightlistReal = [1,1,1.24,1,1,1,1.4,4]
    if t == 'real':
        knnReal(trainingInputPath,testInputPath,k,weightlistReal)
    elif t == 'binary':
        knnBinary(trainingInputPath,testInputPath,k,weightlistBinary)
    else:
        print("invalid type!")

def main(t,action,trainingInputPath,testInputPath):
    k = 5
    if action == 'prediction':
        knn(k,t,trainingInputPath,testInputPath)
    elif action =='validation':
        evaluate(k,t,trainingInputPath,testInputPath)
    else:
        print('invalid argument')

if __name__== "__main__":
    t = sys.argv[1]
    action = sys.argv[2]
    trainingInputPath = sys.argv[3]
    testInputPath = sys.argv[4]
    main(t,action,trainingInputPath,testInputPath)