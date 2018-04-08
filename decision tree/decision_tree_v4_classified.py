import csv
import numpy
import codecs
import math

# Classify continuous feature values
index_cts_features = [2, 3, 7]
min = [0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 5]
max = [0.0, 0.0, 15.00, 7.15, 0.0, 0.0, 0.0, 15]

interval = [0 for i in range(3)]
dictionary = {2:0, 3:1, 7:2}
number_of_cts_featrues = 3
classify_number = 20
print(number_of_cts_featrues)

#for i in range(0, number_of_cts_featrues):
for i in range(len(max)):
    if i not in dictionary:
        continue
    else:
        ind = dictionary.get(i)
        interval[ind] = (int(max[i]) - int(min[i])) / float(classify_number)
print(interval)

def classify_cts_features(dataset):
    pass
    count = 1
    for now_ins in dataset[1:]:
        for i in index_cts_features:
            tmp = float(now_ins[i])
            new_val = (tmp-min[i])
            new_val = int(new_val/interval[dictionary.get(i)])
            now_ins[i] = str(new_val)
            dataset[count][i] = now_ins[i]
        count += 1

    return dataset


def load_data(filename):
    dataset = []
    with codecs.open(filename, "r") as csvfile:
        csvReader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        # dataset = list(csvReader)
        label_list = []
        for row in csvReader:
            # print(row)
            if filename == "csv_result-trainProdIntro-binary.csv":
                label_list.append(row[0].split(',')[9])
                dataset.append(row[0].split(',')[1:9])
            else:
                label_list.append(row[0].split(',')[9])
                dataset.append(row[0].split(',')[1:9])
            # print(row)
        label_list.remove("Label")
        # print(label_list)
        # dataset.append(row)
        # print(dataset)

        dataset = classify_cts_features(dataset)
        return dataset,label_list
        # print(', '.join(row))


# Load a csv file
filename_test = "csv_result-testProdIntro-test-binary.csv"
filename = "csv_result-trainProdIntro-binary.csv"

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

    # print(entropy(attributes, dataset, label))
    # print(sub_entropy)

    return (entropy(attributes, dataset, label) - sub_entropy)


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
    # index = 0
    # # find index of the attribute
    # for index in range(len(attributes)):
    #     if attr == attributes[index]:
    #         break
    # print("index for added attribute is " + str(index))
    values = []
    for row in dataset:
        if row[index] not in values:
            values.append(row[index])
    return values

# Update dataset
def get_newDataset(dataset, attributes, best_attribute, given_val):
    new_dataset = [[]]
    index = attributes.index(best_attribute)
    # for index in range(0, len(attributes)):
    #     if best_attribute == attributes[index]:
    #         break
    # print("index for added attribute is " + str(index))
    for row in dataset:
        # get rows that excludes the given value(e.g. Student)
        if (row[index] == given_val):
            new_entry = []
            # scan the whole row that contains the given_val
            for i in range(0, len(row)):
                if (i != index):
                    new_entry.append(row[i])
            new_dataset.append(new_entry)
    new_dataset.remove([])
    return new_dataset

# Get default value(majority)
def most_majority_label(attributes, dataset, label):
    attribute_val_freq_map = {}
    index = 0
    for index in range(0, len(attributes)):
        if label == attributes[index]:
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


def build_decision_tree(dataset, attributes, label, recurion):
    recurion = recurion + 1
    # Build a new decision tree based on the examples given
    updated_dataset = dataset[:]
    # index = 0
    # for index in range(0, len(attributes)):
    #     if label == attributes[index]:
    #         break
    #print("index for build_decision_tree attribute is " + str(index))
    # Store the label_vals into a label_values list
    label_vals = []
    for row in dataset:
        label_val = row[-1]
        label_vals.append(label_val)
    default = most_majority_label(attributes, dataset, label)
    # Terminate condition
    if not dataset or (len(attributes)) <= 0:
        return default
    elif label_vals.count(label_vals[0]) == len(label_vals):
        return label_vals[0]
    else:
        # Select the best next attributes that can optimal classify dataset
        best_attr = select_best_attribute(dataset, attributes, label)

        # Build a new decision tree with best attribute and
        decision_tree = {best_attr: {}}
        values = get_values(dataset, attributes, best_attr)
        for val in values:
            new_dataset = get_newDataset(dataset, attributes, best_attr, val)
            newAttr = attributes[:]
            newAttr.remove(best_attr)
            subtree = build_decision_tree(new_dataset, newAttr, "Label", recurion)
            # Add the new subtree to the empty dictionary
            decision_tree[best_attr][val] = subtree
    return decision_tree

#
def traversal(node, level):
    # if len(node) == 0:
    if not isinstance(node, dict):
        # print(node)
        return

    for key, _ in node.items():
        traversal(node[key], level+1)

    #for val in node.keys():
     #   traversal(node[val], level + 1)



def inference(data, attributes, dtree):
    if not isinstance(dtree,dict):
        return dtree
    for key, val in dtree.items():
#        print(key)
        index = attributes.index(key)
#        print(index)
        value = data[index]
        if value not in list(val.keys()):
            return "1"
        #     value = min(list(val.keys()), key=lambda x:abs(int(x)-int(value)))
        new_tree = val[str(value)]
        result = inference(data, attributes, new_tree)
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

def main():
    dataset,label_list = load_data(filename)
    dataset_test, label_list_test = load_data(filename_test)

    # print(dataset)

    # Attributes no label column(Y)
    attributes = dataset[0]
    # attributes.remove("Label")

    # dataset includes attributes, no label column
    dataset.remove(attributes)
    dataset_test.remove(attributes)

    # dataset.
    # print(dataset[0:2])
    gain0 = gain(attributes, dataset, "Customer", "Label")
    # gain_test = gain(attributes, dataset, "Outlook", "Play Golf")

    # Run ID3 to build
    # print(gain_test)

    # data_shuffle = random.shuffle(dataset)
    # train_data = data_shuffle[:120]
    # validation = data_shuffle[120:]
    # k_fold = 4
    # cross_validation_accuracy_list = []
    # for X_train, y_train, X_valid, y_valid in k_folder_generator(dataset, label_list, k_fold):
    #     dtree = build_decision_tree(X_train, attributes, "Label", 0)
    #     print(dtree.keys())
    #     traversal(dtree, 0)
    #     prediction = []
    #
    #     for data in X_valid:
    #         prediction.append(inference(data, attributes, dtree))
    #
    #     count = 0
    #     for i in range(len(prediction)):
    #         if prediction[i] == y_valid[i]:
    #             count += 1
    #     print("Accuracy for cross-validation data is " + str(count / len(prediction)))
    #     cross_validation_accuracy_list.append(str(count / len(prediction)))

    dtree = build_decision_tree(dataset, attributes, "Label", 0)
    print(dtree)
    traversal(dtree, 0)
    #
    prediction_test = []
    #
    for data in dataset_test:
        prediction_test.append(inference(data, attributes, dtree))
    #
    count = 0
    for i in range(len(prediction_test)):
        if prediction_test[i] == label_list_test[i]:
            count += 1
    print("Accuracy for test data is " + str(count/len(prediction_test)))
if __name__ == '__main__':
    main()