import random
from knnB_binary import *
import numpy as np

def generate_random_weights():
	weights=[]
	for i in range(8):
		weights.append(random.uniform(0,10))
	return weights

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

def optimize_weights(k):
	datasets = n_fold("./trainProdIntro_binary.csv",10)
	global_max_accuracy = 0
	global_weight_list = []
	for t in range(120):
		tmp_accuracy = 0
		weightlist = generate_random_weights()
		for i in range(len(weightlist)):
			while weightlist[i] < 10:
				accuracy = get_average_accuracy(datasets,k,weightlist)
				if accuracy < tmp_accuracy:
					break
				tmp_accuracy = accuracy
				weightlist[i] += 0.03
		if tmp_accuracy > global_max_accuracy:
			global_max_accuracy = tmp_accuracy
			global_weight_list = weightlist
			print("tmp_accuracy: ",tmp_accuracy," weightlist: ",normalize_weights(weightlist))
	print(normalize_weights(global_weight_list),global_max_accuracy)
	return global_weight_list

def optimize_weights_tmp(k):
	datasets = n_fold("./trainProdIntro_binary.csv",10)
	global_weight_list = []
	global_max_accuracy = 0
	for t in range(20):
		weightlist = generate_random_weights()
		dic = {}
		for i in range(len(weightlist)):
			best_acc = 0
			best_weight_value = 0
			for j in np.arange(weightlist[i],10,0.3):
				weightlist[i] = j
				accuracy = get_average_accuracy(datasets,k,weightlist)
				if accuracy > best_acc:
					best_acc = accuracy
					best_weight_value = j
			dic[i] = best_weight_value
		local_weights = []
		for k in range(len(dic)):
			local_weights.append(dic[k])
		local_acc = get_average_accuracy(datasets,k,local_weights)
		if local_acc > global_max_accuracy:
			global_max_accuracy = local_acc
			global_weight_list = local_weights
		print("local_weights",local_weights)
		print("local_acc",local_acc)
	print(global_weight_list,global_max_accuracy)
	return global_weight_list

def get_average_accuracy(datasets,k,weightlist):
	sum = 0
	for i in range(10):
		set = datasets[i]
		training_set = set[0]
		testing_set = set[1]
		acc = getAccuracy(training_set,testing_set,k,weightlist)
		sum += acc
	return sum/10

def normalize_weights(weightlist):
	sum = 0
	for i in range(len(weightlist)):
		sum +=weightlist[i]
	for i in range(len(weightlist)):
		weightlist[i] = weightlist[i]/sum
	return weightlist

def test_accuracy():
	datasets = n_fold("./trainProdIntro_binary.csv",10)
	weightlist=[0.014461023712459402, 0.14834202150492865, 0.010759602207293533, 0.13509735230635024, 0.07996645391153464, 0.0683162827997523, 0.26068668442722375, 0.29337057913045744]
	acc = get_average_accuracy(datasets,5,weightlist)
	print(acc)

#test_accuracy()
optimize_weights(5)
