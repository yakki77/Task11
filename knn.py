import math
import operator

#wait to be filled
def loadData():


def euclidDistance(vector1,vector2):
	length = len(vector1)
	distance = 0;
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

main()

