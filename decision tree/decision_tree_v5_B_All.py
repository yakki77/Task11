import csv
import numpy
import codecs
import math
import random
from collections import Counter

# Load a csv file
filename = "csv_result-trainProdIntro_B_binary.csv"
filename_test = "csv_result-testProdIntro_B_binary.csv"

filename_train_B_real = "csv_result-trainProdIntro_B_real.csv"
filename_test_B_real = "csv_result-testProdIntro_B_real.csv"

filename_train_A = "csv_result-trainProdSelection_A.csv"
filename_test_A = "csv_result-testProdSelection_A.csv"

# Split Discrete dataset to get all feature values
def split_discrete_dataset(dataset, axis, value):
    result = []
    for row in dataset:
        if row[axis] == value:
            reduceRow = row[:axis]
            reduceRow.extend(row[axis + 1:])
            result.append(reduceRow)
    return result

# Split Continuous dataset (decide to divide < value / > value)
def split_continuous_dataset(dataset, axis, value, direction):
    result_dataset = []
    for row in dataset:
        if direction == 0:
            if float(row[axis]) > value:
                reduced_featureVec = row[:axis]
                reduced_featureVec.extend(row[axis + 1:])
                result_dataset.append(reduced_featureVec)
            else:
                if float(row[axis]) <= value:
                    reduced_featureVec = row[:axis]
                    reduced_featureVec.extend(row[axis + 1:])
                    result_dataset.append(reduced_featureVec)
    return result_dataset

# Select the best way to divide the dataset
def select_best_feature_to_split(dataset, attributes):
    # number_features = len(attributes) # 8
    base_entropy = calc_shannon_entropy(dataset)
    best_gain = 0.0
    best_feature = -1
    best_split_dict = {}
    for i in range(len(attributes)):
        feature_list = [row[i] for row in dataset] # get the column of attribute (for index i)
        # Deal with continuous features
        if isinstance(feature_list[0],float) or isinstance(feature_list[0],int):
            feature_list = list(map(float, feature_list))
            # print(feature_list)
            sorted_feature_list = sorted(feature_list)
            split_list = []
            for j in range(len(sorted_feature_list) - 1):
                split_list.append((sorted_feature_list[j] + sorted_feature_list[j + 1]) / 2.0)
            best_split_entropy = 10000
            split_list_len = len(split_list)

            # the j-th divide point to divide, get the entropy, and recored the best divide point
            for j in range(split_list_len):
                value = split_list[j]
                new_entropy = 0.0
                sub_dataset0 = split_continuous_dataset(dataset, i, value, 0)
                sub_dataset1 = split_continuous_dataset(dataset, i, value, 1)
                prob_0 = len(sub_dataset0) / float(len(dataset))
                new_entropy += prob_0 * calc_shannon_entropy(sub_dataset0)
                prob_1 = len(sub_dataset1) / float(len(dataset))
                new_entropy += prob_1 * calc_shannon_entropy(sub_dataset1)
                if new_entropy < best_split_entropy:
                    best_split_entropy = new_entropy
                    best_split_point = j
            # use dictionary to record the best divided point
            best_split_dict[attributes[i]] = split_list[best_split_point]
            info_gain = base_entropy - best_split_entropy
        # Deal with discrete features
        else:
            unique_val = set(feature_list)
            new_entropy = 0.0
            # Calculate the entropy to divide the feature
            for value in unique_val:
                sub_dataset = split_discrete_dataset(dataset, i, value)
                prob = len(sub_dataset) / float(len(dataset)) # get prob = student / total
                new_entropy += prob * calc_shannon_entropy(sub_dataset)
            info_gain = base_entropy - new_entropy

        if info_gain > best_gain:
            best_gain = info_gain
            best_feature = i

    if type(dataset[0][best_feature]) == float or type(dataset[0][best_feature]) == int:
        best_split_val = best_split_dict[attributes[best_feature]]
        # attributes[best_feature] = attributes[best_feature] + "<=" + str(best_split_val)
        for i in range(len(dataset)):
            if float(dataset[i][best_feature]) <= best_split_val:
                dataset[i][best_feature] = 1
            else:
                dataset[i][best_feature] = 0
    return best_feature

# Calculate entropy
def calc_shannon_entropy(dataset):
    number_of_entries = len(dataset)
    label_counts = {}
    # build dictionary for all features
    for feaVec in dataset:
        curr_label = feaVec[-1]
        if curr_label not in label_counts.keys():
            label_counts[curr_label] = 0
        label_counts[curr_label] += 1
    shannon_ent = 0.0
    # Calculate shannon_en
    for key in label_counts:
        prob = float(label_counts[key]) / number_of_entries
        shannon_ent -= prob * math.log(prob, 2)
    return shannon_ent


def load_data(filename):
    dataset = []
    with codecs.open(filename, "r") as csvfile:
        csvReader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        # dataset = list(csvReader)
        label_list = []
        label_list_B_real = []
        for row in csvReader:
            if filename == "csv_result-trainProdIntro_B_binary.csv" or\
                            filename == "csv_result-testProdIntro_B_binary.csv":
                label_list.append(row[0].split(',')[-1])
                dataset.append(row[0].split(',')[1:-1])
            elif filename == "csv_result-trainProdSelection_A.csv" or \
                            filename == "csv_result-testProdSelection_A.csv":
                label_list.append(row[0].split(',')[-1])
                dataset.append(row[0].split(',')[1:-1])
            elif filename == "csv_result-trainProdIntro_B_real.csv" or \
                            filename == "csv_result-testProdIntro_B_real.csv":
                dataset.append(row[0].split(',')[1:-1])
                label_list_B_real.append(row[0].split(',')[-1])
                if "Label" in label_list_B_real:
                    label_list_B_real.remove("Label")
                label_list = [1 if float(x) >= 20.00 else 0 for x in label_list_B_real]
        if "label" in label_list:
            label_list.remove("label")
        elif "Label" in label_list:
            label_list.remove("Label")
        return dataset,label_list


# Calculate entropy
def entropy(attributes, dataset, target_X):
    attribute_val_freq_map = {}
    entropy = 0.0
    # index = attributes.index(target_X)
    index = 0
    # find the index of the target attribute
    for index in range(len(attributes)):
        if target_X == attributes[index]:
            break

    # Sum up the frequency of each of the value in target attribute X
    for row in dataset:
        # print(row)
        value = row[index]
        if value not in attribute_val_freq_map.keys():
            attribute_val_freq_map[value] = 1.0
        else:
            attribute_val_freq_map[value] += 1.0

    # Calculate the entropy of the data for target
    # E(S) = Sum (-Pi) * log(Pi)
    for freq in attribute_val_freq_map.values():
        entropy += (-freq / len(dataset)) * math.log(freq / len(dataset), 2)
    return entropy


# Gain(T, X) = Entropy(T) - Entropy(T, X)
def gain(attributes, dataset, attribute, label):
    attribute_val_freq_map = {}
    sub_entropy = 0.0
    index = attributes.index(attribute)

    # Sum the frequency of each of the values in target attributes
    for row in dataset:
        value = row[index]
        if value not in attribute_val_freq_map.keys():
            attribute_val_freq_map[value] = 1.0
        else:
            attribute_val_freq_map[value] += 1.0
    # print(attribute_val_freq_map)

    # Calculate the sum
    # E(T, X) = Sum P(a) * E(a)
    for k in attribute_val_freq_map.keys():
        prob_k = attribute_val_freq_map[k] / sum(attribute_val_freq_map.values())
        datasubset = []
        for row in dataset:
            if row[index] == k:
                datasubset.append(row)
        temp = entropy(attributes, datasubset, label)
        sub_entropy += prob_k * temp

    return (entropy(attributes, dataset, label) - sub_entropy) / entropy(attributes, dataset, label)


# Select the best attribute
def select_best_attribute(dataset, attributes, label):
    best_attr = attributes[0]
    max_gain = 0
    for attr in attributes:
        gain_latest = gain(attributes, dataset, attr, label)
        if gain_latest > max_gain:
            max_gain = gain_latest
            best_attr = attr
    return best_attr


# get values in the attribute, such as "Student", "Business", "Professional"
def get_values(dataset, attributes, attr):
    index = attributes.index(attr)
    values = []
    for row in dataset:
        if row[index] not in values:
            values.append(row[index])
    return values

# Update dataset
def get_newDataset(dataset, attributes, best_attribute, given_val):
    new_dataset = [[]]
    row_number_list = []
    index = attributes.index(best_attribute)
    row_number = 0
    for row in dataset:
        # get rows that excludes the given value(e.g. Student)
        if (row[index] == given_val):
            new_entry = []
            # scan the whole row that contains the given_val
            for i in range(0, len(row)):
                if (i != index):
                    new_entry.append(row[i])
            row_number_list.append(row_number)
            new_dataset.append(new_entry)
        row_number =row_number+1
    new_dataset.remove([])
    return new_dataset, row_number_list

# Get default value(majority)
def most_majority_label(attributes, label_list, label):
    attribute_val_freq_map = {}
    index = 0
    for index in range(0, len(attributes)):
        if label == attributes[index]:
            break
    # print("index for build_decision_tree attribute is " + str(index))

    # Sum the frequency of each of the values in target attributes
    for row in label_list:
        value = row[index]
        if value not in attribute_val_freq_map.keys():
            attribute_val_freq_map[value] = 1.0
        else:
            attribute_val_freq_map[value] += 1.0
    max_cnt = 0
    most_common = ""
    for k in attribute_val_freq_map.keys():
        if attribute_val_freq_map[k] > max_cnt:
            max_cnt = attribute_val_freq_map[k]
            most_common = k
    return most_common


def most_majority_val(attributes, dataset, key):
    attribute_val_freq_map = {}
    index = 0
    for index in range(0, len(attributes)):
        if key == attributes[index]:
            break
    # print("index for build_decision_tree attribute is " + str(index))

    # Sum the frequency of each of the values in target attributes
    for row in dataset:
        value = row[index]
    if value not in attribute_val_freq_map.keys():
        attribute_val_freq_map[value] = 1.0
    else:
        attribute_val_freq_map[value] += 1.0
    max_cnt = 0
    most_common = ""
    for k in attribute_val_freq_map.keys():
        if attribute_val_freq_map[k] > max_cnt:
            max_cnt = attribute_val_freq_map[k]
            most_common = k
    return most_common


def majority_count(prediction_list):
    prediction_label_cnt = {}
    for vote in prediction_list:
        if vote not in prediction_label_cnt.keys():
            prediction_label_cnt[vote] = 0
        try:
            prediction_label_cnt[vote] += 1
        except:
            print("error")
    return max(prediction_label_cnt)


def build_decision_tree(dataset, label_list, attributes, label, recursion, depth):
    recursion = recursion + 1
    # Build a new decision tree based on the examples given
    updated_dataset = dataset[:]
    if label_list.count(label_list[0]) == len(label_list):
        return label_list[0]

    if len(dataset[0]) == 1:
        return majority_count(label_list)

    default = most_majority_label(attributes, dataset, label)
    # Terminate condition
    if not dataset or (len(attributes)) <= 0:
        return default
    elif label_list.count(label_list[0]) == len(label_list):
        return label_list[0]
    else:
        # Select the best next attributes that can optimal classify dataset
        # best_attr = select_best_attribute(dataset, attributes, label)
        best_attr_idx = select_best_feature_to_split(dataset, attributes)
        best_attr = attributes[best_attr_idx]

        # Build a new decision tree with best attribute and
        decision_tree = {best_attr: {}}
        for row in dataset:
            feature_val = row[best_attr_idx]
       # unique_val = set(feature_val)

        values = get_values(dataset, attributes, best_attr)
        for val in values:
            new_dataset, row_number_list = get_newDataset(dataset, attributes, best_attr, val)
            select_label_list=[label_list[x] for x in row_number_list]
            newAttr = attributes[:]
            newAttr.remove(best_attr)
            if depth <= 5:
                subtree = build_decision_tree(new_dataset, select_label_list, newAttr, "Label", recursion, depth + 1)
                # Add the new subtree to the empty dictionary
                decision_tree[best_attr][val] = subtree
            else:
                decision_tree[best_attr][val] = subtree
            # Add the new subtree to the empty dictionary
            # decision_tree[best_attr][val] = subtree
    if "0" in label_list or "1" in label_list:
        label_0_count=label_list.count("0")
        label_1_count=label_list.count("1")
        max_count_label="1"
        if label_0_count>label_1_count:
            max_count_label="0"
        decision_tree['default']=max_count_label

    if "C1" in label_list or "C2" in label_list or "C3" in label_list or "C4" in label_list or "C5" in label_list:
        count_label = Counter(label_list)
        most_common = count_label.most_common()[0]
        max_count_label_for_A = most_common[0]
        # print("max_count_label" + max_count_label_for_A)
        decision_tree['default_A'] = max_count_label_for_A
    return decision_tree

# Traverse the decision tree
def traversal(node, level):
    # if len(node) == 0:
    if not isinstance(node, dict):
        # print(node)
        return

    for key, _ in node.items():
        traversal(node[key], level+1)


def inference(data, attributes, dtree):
    if not isinstance(dtree,dict):
        return dtree
    for key, val in dtree.items():
        if key=="default":
            continue
        index = attributes.index(key)
        value = data[index]
        if value not in list(val.keys()):
            # return majority_count()
            return dtree['default']
        #     value = min(list(val.keys()), key=lambda x:abs(int(x)-int(value)))
        new_tree = val[str(value)]
        result = inference(data, attributes, new_tree)
    return result


def inference_A(data, attributes, dtree):
    if not isinstance(dtree,dict):
        return dtree
    for key, val in dtree.items():
        if key=="default_A":
            continue
        index = attributes.index(key)
        value = data[index]
        if value not in list(val.keys()):
            # return majority_count()
            return dtree['default_A']
        #     value = min(list(val.keys()), key=lambda x:abs(int(x)-int(value)))
        new_tree = val[str(value)]
        result = inference_A(data, attributes, new_tree)
    return result

def k_folder_generator(dataset,label_list, k_folder_number):
    unit_size = (int)(len(dataset) / k_folder_number) # cast to int
    print(unit_size)
    for k in range(k_folder_number):
        subset_size = k * unit_size # 0*40, 1*40, 2*40, 3*40,
        X_valid = dataset[subset_size:subset_size + unit_size]
        X_train = dataset[:subset_size] + dataset[subset_size + unit_size:]
        y_valid = label_list[subset_size : subset_size + unit_size]
        y_train = label_list[:subset_size] + label_list[subset_size + unit_size:]
        yield X_train, y_train, X_valid, y_valid

def convert_to_numerical_feature(dataset):
    ret_dataset=[]
    cnt=0
    for now_row in dataset:
        cnt+=1
        if cnt==1:
            ret_dataset.append(now_row)
            continue
        new_row=[]
        for ele in now_row:
            ele_float=None
            try:
                ele_float=float(ele)
            except:
                ele_float=ele
            new_row.append(ele_float)

        ret_dataset.append(new_row)
        pass
    return ret_dataset
def main():
    #################################
    ## Load data for dataset B binary
    # #################################
    # dataset,label_list = load_data(filename)
    # dataset_test, label_list_test = load_data(filename_test)

    #################################
    ## Load data for dataset B real
    #################################
    dataset,label_list = load_data(filename_train_B_real)
    dataset_test, label_list_test = load_data(filename_test_B_real)

    #################################
    ##   Load data for dataset A
    #################################
    # dataset,label_list = load_data(filename_train_A)
    # dataset=convert_to_numerical_feature(dataset)


    # dataset_test, label_list_test = load_data(filename_test_A)
    # dataset_test=convert_to_numerical_feature(dataset)

    # Attributes no label column(Y)
    attributes = dataset[0]
    attributes_full = attributes[:]

    # dataset includes attributes, no label column
    dataset.remove(attributes)

    # Run ID3 to build
    data_shuffle = random.shuffle(dataset)
    k_fold = 4
    cross_validation_accuracy_list = []
    if "0" in label_list or "1" in label_list:
        for X_train, y_train, X_valid, y_valid in k_folder_generator(dataset, label_list, k_fold):
            dtree = build_decision_tree(X_train, y_train, attributes, "Label", 0, 0)
            # print(dtree.keys())
            traversal(dtree, 0)
            prediction = []
            for data in X_valid:
                prediction.append(float(inference(data, attributes, dtree)))
            print("prediction for cross validation " + str(prediction))
            count = 0
            for i in range(len(prediction)):
                if prediction[i] == float(y_valid[i]):
                    count += 1
            print("Accuracy for cross-validation data is " + str(count / len(prediction)))
            cross_validation_accuracy_list.append(str(count / len(prediction)))


        # Test for B (Binary / Real)
        dtree = build_decision_tree(dataset, label_list, attributes, "Label", 0, 0)
        print("The decision tree: " + str(dtree))
        traversal(dtree, 0)
        #
        prediction_test = []
        #
        for data in dataset_test:
            prediction_test.append(inference(data, attributes, dtree))

        count_for_zero = 0
        for pred in prediction_test:
            if pred == 0 or pred == '0':
                print("index for zero in test prediction:" + str(count_for_zero))
            count_for_zero = count_for_zero + 1
        print("prediction " + str(prediction_test))

    if "C1" in label_list or "C2" in label_list or "C3" in label_list or "C4" in label_list or "C5" in label_list:
        for X_train, y_train, X_valid, y_valid in k_folder_generator(dataset, label_list, k_fold):
            dtree = build_decision_tree(X_train, y_train, attributes, "Label", 0, 0)
            # print(dtree.keys())
            traversal(dtree, 0)
            prediction = []

            for data in X_valid:
                prediction.append(inference_A(data, attributes, dtree))
            print("prediction for cross validation " + str(prediction))
            count = 0
            for i in range(len(prediction)):
                if prediction[i] == (y_valid[i]):
                    count += 1
            print("Accuracy for cross-validation data is " + str(count / len(prediction)))
            cross_validation_accuracy_list.append(str(count / len(prediction)))

if __name__ == '__main__':
    main()