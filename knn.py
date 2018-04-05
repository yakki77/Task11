import math
import operator
import csv
import numpy as np

class Person():
    def __init__(self, personal_type, life_style, vacation, e_credit, salary, p_property):
        self.personal_type = personal_type
        self.life_style = life_style
        self.vacation = vacation
        self.e_credit = e_credit
        self.salary = salary
        self.p_property = p_property

    def print_person(self):
        print "type:"
        print self.personal_type
        print "style:"
        print self.life_style
        print "vacation: {}, eCreadit: {}, salary:{}, property:{}".format(self.vacation,
         self.e_credit, self.salary, self.p_property)

def typeMatrix(row):
    matrix = np.zeros((5,5))
    if row == 'student':
        matrix[0,0] = 1.0
    elif row == 'engineer':
        matrix[1,1] = 1.0
    elif row == 'librarian':
        matrix[2,2] = 1.0
    elif row == 'professor':
        matrix[3,3] = 1.0
    else:
        matrix[4,4] = 1.0
    return matrix

def lifeStyleMatrix(row):
    matrix = np.zeros((4,4))
    if row == 'spend<<saving':
        matrix[0,0] = 1.0
    elif row == 'spend<saving':
        matrix[1,1] = 1.0
    elif row == 'spend>saving':
        matrix[2,2] = 1.0
    else:
        matrix[3,3] = 1.0
    return matrix

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
    va_list = [float(i) for i in testSet[:,3]]
    va_list.sort()

    creadit_list = [float(i) for i in testSet[:,4]]
    creadit_list.sort()

    salary_list = [float(i) for i in testSet[:,5]]
    salary_list.sort()

    property_list = [float(i) for i in testSet[:,6]]
    property_list.sort()

    for i in range(len(testSet[:,0])):
        p_type = testSet[:,1][i]
        p_type_matrix = typeMatrix(p_type)

        p_life_style = testSet[:,2][i]
        p_life_style_matric = lifeStyleMatrix(p_life_style)

        va = count(va_list[-1], va_list[0], testSet[i,3])
        credit = count(creadit_list[-1], creadit_list[0], testSet[i,4])
        salary = count(salary_list[-1], salary_list[0], testSet[i,5])
        property_ = count(property_list[-1], property_list[0], testSet[i,6])

        person = Person(p_type_matrix, p_life_style_matric, va, credit, salary, property_)
        person_list.append(person)
    
    for p in person_list:
        p.print_person()


def count(max_num, min_num, num):
    num = float(num)
    return (num - min_num) / (max_num - min_num)



def euclidDistance(vector1,vector2):
	length = len(vector1)
	distance = 0
	for index in range(length):
		distance = distance + pow(vector1[index] - vector2[index],2)
	return math.sqrt(distance)

def findNeighborList(testVector,trainningSet,k):
	neighbors = []
	length = len(trainningSet)
	for index in range(length):
		distance = euclidDistance(trainningSet[index],testVector)
		neighbors.append(trainningSet[index],distance)
	neighbors.sort(key=operator.itemgetter(1))
	ans = []
	for neighbor in range(k):
		ans.append(neighbors[x][0])
	return ans

def getLabel(neighbors):
	frequency={}
	length = len(neighbors)
	for index in range(length):
		label= neighbors[index][-1]
		if label in frequency:
			frequency[label] = frequency[neighbor] + 1
		else:
			frequency[label] = 1
	return sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)[0][0]

def main():
	trainningSet=[]
	testSet=[]
	# This method should fill trainningSet and testSet
	loadData()
	k = 8
	predictions = []
	length = len(testSet)
	for i in range(length):
		neighbors = findNeighborList(testSet[i],trainningSet,8)
		label = getLabel(neighbors)
		predictions.append(label)
	print(predictions)

loadData("trainProdSelection.csv")

