import random
from knnB_real import *

def generate_random_weights():
	weights=[]
	for i in range(8):
		weights.append(random.uniform(0,10))
	return weights

def n_fold():


def optimize_weights(trainningInputPath,testInputPath,k):
	global_min_mse = 100000
	global_weight_list = []
	for t in range(80):
		tmp_mse = 100000
		weightlist = generate_random_weights()
		print("initial: ",weightlist)
		for i in range(len(weightlist)):
			while weightlist[i] < 10:
				mse = getAccuracy(trainningInputPath,testInputPath,k,weightlist)
				if mse > tmp_mse:
					weightlist[i] -= 0.01
					break
				tmp_mse = mse
				weightlist[i] += 0.01
		if tmp_mse < global_min_mse:
			global_min_mse = tmp_mse
			global_weight_list = weightlist
			print("tmp_mse: ",tmp_mse," weightlist: ",weightlist)
	return global_weight_list

def main():
	weightlist = optimize_weights('./trainProdIntro_real.csv','./trainProdIntro_real.csv',5)
	final_mse = getAccuracy('./trainProdIntro_real.csv','./trainProdIntro_real.csv',5,weightlist)
	print("final answer: ",final_mse)
	print("weightlist: ",weightlist)

main()


