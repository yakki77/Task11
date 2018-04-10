import random
from knnB_real import *

def generate_random_weights():
	weights=[]
	for i in range(8):
		weights.append(random.uniform(0,1))
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
	datasets = n_fold("./trainProdIntro_real.csv",10)
	global_min_mse = 100000
	global_weight_list = []
	for t in range(60):
		tmp_mse = 100000
		weightlist = generate_random_weights()
		for i in range(len(weightlist)):
			while weightlist[i] < 10:
				mse = get_average_mse(datasets,k,weightlist)
				if mse > tmp_mse:
					weightlist[i] -= 0.08
					break
				tmp_mse = mse
				weightlist[i] += 0.08
		if tmp_mse < global_min_mse:
			global_min_mse = tmp_mse
			global_weight_list = weightlist
			print("tmp_mse: ",tmp_mse," weightlist: ",weightlist)
	print(normalize_weights(global_weight_list))
	return normalize_weights(global_weight_list)

def normalize_weights(weightlist):
	sum = 0
	for i in range(len(weightlist)):
		sum +=weightlist[i]
	for i in range(len(weightlist)):
		weightlist[i] = weightlist[i]/sum
	return weightlist

def test_accuracy():
	datasets = n_fold("./trainProdIntro_real.csv",10)
	weightlist=[0.014461023712459402, 0.14834202150492865, 0.010759602207293533, 0.13509735230635024, 0.07996645391153464, 0.0683162827997523, 0.26068668442722375, 0.29337057913045744]
	acc = get_average_mse(datasets,5,weightlist)
	print(acc)

optimize_weights(5)


