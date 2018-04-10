import math
import operator
import csv

def loadData(trainingInputPath, testInputPath):
    set = []
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
    for index in range(length):
    	label= neighbors[index][-1]
    	if label in frequency:
    		frequency[label] += 1
    	else:
    		frequency[label] = 1
    ans = sorted(frequency.items(), key=operator.itemgetter(1), reverse=True)
    return ans[0][0]

def knn(trainningInputPath,testInputPath,k,weightlist):
    set = loadData(trainningInputPath,testInputPath)
    trainingSet = set[0]
    testSet = set[1]
    predictions = []
    length = len(testSet)
    for i in range(length):
        neighbors = findNeighborList(testSet[i],trainingSet,k,weightlist)
        label = getLabel(neighbors)
        predictions.append(label)
    print(predictions)

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
        set.append(training_set)
        set.append(testing_set)
        datasets[i] = set 
    return datasets

def get_average_accuracy(datasets,k,weightlist):
    sum = 0
    for i in range(10):
        set = datasets[i]
        training_set = set[0]
        testing_set = set[1]
        acc = getAccuracy(training_set,testing_set,k,weightlist)
        sum += acc
    return sum/10

def evaluate(trainingInputPath,k,weightlist):
    datasets = n_fold(trainingInputPath,10)
    acc = get_average_accuracy(datasets,k,weightlist)
    print(acc*100,"%")

def main():
    k = 5;
    testInputPath = './testProdIntro_binary.csv'
    trainingInputPath = './trainProdIntro_binary.csv'
    weightlist = [0.024461023712459402, 0.15834202150492865, 0.010759602207293533, 0.10509735230635024, 0.08996645391153464, 0.0683162827997523, 0.25068668442722375, 0.29237057913045744]
    #knn(trainingInputPath,testInputPath,k,weightlist)
    evaluate(trainingInputPath,k,weightlist)

main()

