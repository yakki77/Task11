import csv
import math
import numpy as np
import sys

def load_data(trainningInputPath, testInputPath=None):
    trainingSet = []
    with open(trainningInputPath, "r") as csvfile:
        test_data = csv.reader(csvfile, delimiter=' ', quotechar='|')
        index = -1
        for row in test_data:
            if index >= 0:
                features = row[0].split(',')
                trainingSet.append(features)
            index += 1
    if testInputPath != None:
        testSet = []
        with open(testInputPath, "r") as csvfile:
            test_data = csv.reader(csvfile, delimiter=' ', quotechar='|')
            index = -1
            for row in test_data:
                if index >= 0:
                    features = row[0].split(',')
                    testSet.append(features)
                index += 1
        return trainingSet, testSet
    else:
        return trainingSet

def entropy(dataset, feature_index):
    E = 0
    number_of_entries = len(dataset)
    feature_type = {}
    for i in range(number_of_entries):
        each = dataset[i][feature_index]
        if each in feature_type:
            feature_type[each] += 1
        else:
            feature_type[each] = 1
    for num in feature_type.values():
        prob = num/number_of_entries
        E += -prob * math.log(prob, 2)

    return E

def split_data(dataset, feature_index):
    features = [row[feature_index] for row in dataset ]
    unique_feature = list(set(features))
    sub_data = {}
    for each in unique_feature:
        sub_data[each] = []
        for row in dataset:
            if row[feature_index] == each:
                sub_data[each].append(row[:feature_index] + row[feature_index+1:])
    return sub_data


class TreeNode:
    def __init__(self, feature_index):
        self.feature_index = feature_index
        self.children = {}

def createTree(dataset):
    if len(dataset[0]) == 1:
        return getMajority(dataset)
    if len(set([row[-1] for row in dataset])) == 1:
        return str(dataset[0][-1])

    curr_feature = None
    maxIG_ratio = 0
    for i in range(len(dataset[0])-1):
        r = IG_ratio(dataset, i)
        if r > maxIG_ratio:
            maxIG_ratio = r
            curr_feature = i
    if curr_feature is None:
        return getMajority(dataset)

    sub_datasets = split_data(dataset, curr_feature)
    node = TreeNode(curr_feature)
    for k,v in sub_datasets.items():
        node.children[k] = createTree(v)
    return node

def IG_ratio(dataset, feature_index):
    E_parent = entropy(dataset, -1)
    E_feature = entropy(dataset, feature_index)
    if E_feature != 0:
        sub_dataset = split_data(dataset, feature_index)

        E_child = 0
        for sub_data in sub_dataset.values():
            E_sub = entropy(sub_data, -1)
            E_child += E_sub*(len(sub_data)/len(dataset))
        IG = E_parent - E_child
        IG_ratio = IG/E_feature
    else:
        IG_ratio = 0
    return IG_ratio

def getMajority(dataset):
    if isNum(dataset[0][-1]) == True:
        labels = [row[-1] for row in dataset]
        label_counts = np.bincount(labels)
        return str(np.argmax(label_counts))
    else:
        labels = [row[-1][-1] for row in dataset]
        label_counts = np.bincount(labels)
        return 'C' + str(np.argmax(label_counts))

def isNum(column):
    try:
        float(column[0])
        return True
    except ValueError:
        return False

def isContinuous(column):
    unique_data = set(column)
    if len(unique_data) > 10:
        return True
    return False

def split_continuous(column, division = 5):
    column_max = max(column)
    column_min = min(column)
    step = (column_max - column_min)/ float(division)
    res = column
    flag = [False] * len(column)
    for i in range(division):
        for j in range(len(column)):
            if i == 0:
                if column[j] >= column_min and column[j] <= (i+1) * step + column_min:
                    if not flag[j]:
                        res[j] = i
                        flag[j] = True
            else:
                if column[j] > column_min and column[j] <= (i+1) * step + column_min:
                    if not flag[j]:
                        res[j] = i
                        flag[j] = True

    return res

def process_data(train, test):
    dataset = train + test
    for i in range(len(dataset[0])-1):
        column = [row[i] for row in dataset]
        if isNum(column) == True:
            if isContinuous(column) == True:
                column_f = [float(i) for i in column]
                column_cate = split_continuous(column_f)
                for j in range(len(dataset)):
                    dataset[j][i] = column_cate[j]
    clean_train = dataset[0:len(train)]
    clean_test = dataset[len(train):]
    return clean_train, clean_test
    

def classify(tree, data_row):
    feature = tree.feature_index
    for key, value in tree.children.items():
            if data_row[feature] == key:
                if type(value) is str:
                    return value
                else:
                    tree = value
                    data_row = data_row[:feature] + data_row[feature+1:]
                    return classify(tree, data_row)

def cross_validation(data, n=1):
    count = 0
    length = len(data)
    for i in range(int((length-1) / n) + 1):
        test = data[i * n : (i  + 1) * n]
        train = data[0:i * n] + data[(i + 1) * n:length]
        clean_train, clean_test = process_data(train, test)

        tree = createTree(clean_train)
        for row in clean_test:
            if row[-1] == classify(tree, row):
                count += 1

    return count/length

def main(action, train, test):
    if action == 'classify':
        train_data, test_data = load_data(train, test)
        clean_train, clean_test = process_data(train_data, test_data)
        if isNum(clean_train[0][-1]):
            default = '1'
        else:
            default = 'C1'
        tree = createTree(clean_train)
        labels = []
        for row in clean_test:
            l = classify(tree, row)
            if l is None:
                labels.append(default)
            else:
                labels.append(l)
        print(labels)
    else:
        train_data = load_data(train)
        print(cross_validation(train_data))


if __name__ == "__main__":
	action = sys.argv[1]
	train = sys.argv[2]
	test = sys.argv[3]
	main(action, train, test)
	



