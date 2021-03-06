import csv
import numpy
import codecs
import math

def load_data(filename):
    dataset = []
    with codecs.open(filename, "r") as csvfile:
        csvReader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        # dataset = list(csvReader)
        label_list = []
        for row in csvReader:
            print(row)
            label_list.append(row[0].split(',')[9])
            dataset.append(row[0].split(',')[1:- 2])
        label_list.remove("Label")
        # dataset.append(row)
        # print(dataset)
        return dataset,label_list
        # print(', '.join(row))


# Load a csv file
filename = "csv_result-trainProdIntro-binary.csv"
# filename = "test.csv"


# dataset = load_data(filename)
# print(dataset[0])

# Calculate
def entropy(attributes, dataset, target_X):
    attribute_val_freq_map = {}
    entropy = 0.0

    index = 0
    # find the index of the target attribute
    for index in range(len(attributes)):
        if target_X == attributes[index]:
            break
    print("index for target attribute is " + str(index))

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

    index = 0
    # find index of the attribute
    for index in range(len(attributes)):
        if attribute == attributes[index]:
            break
    print("index for added attribute is " + str(index))

    # Sum the frequency of each of the values in target attributes
    for row in dataset:
        value = row[index]
        if value not in attribute_val_freq_map.keys():
            attribute_val_freq_map[value] = 1.0
        else:
            attribute_val_freq_map[value] += 1.0
    print(attribute_val_freq_map)

    # Calculate the sum
    # E(T, X) = Sum P(a) * E(a)
    for k in attribute_val_freq_map.keys():
        prob_k = attribute_val_freq_map[k] / sum(attribute_val_freq_map.values())
        print(prob_k)
        datasubset = []
        for row in dataset:
            if row[index] == k:
                datasubset.append(row)
        temp = entropy(attributes, datasubset, label)
        print(temp)
        sub_entropy += prob_k * temp

    print(entropy(attributes, dataset, label))
    print(sub_entropy)

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
    index = 0
    # find index of the attribute
    for index in range(len(attributes)):
        if attr == attributes[index]:
            break
    print("index for added attribute is " + str(index))
    values = []
    for row in dataset:
        if row[index] not in values:
            values.append(row[index])
    return values


def get_newDataset(dataset, attributes, best_attribute, given_val):
    new_dataset = [[]]
    for index in range(0, len(attributes)):
        if best_attribute == attributes[index]:
            break
    print("index for added attribute is " + str(index))
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


def most_majority_val(attributes, dataset, label):
    attribute_val_freq_map = {}
    index = 0
    for index in range(0, len(attributes)):
        if label == attributes[index]:
            break
    print("index for build_decision_tree attribute is " + str(index))

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
        max_ = max(attribute_val_freq_map[k], max_cnt)
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
    default = most_majority_val(attributes, dataset, label)
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


def traversal(node, level):
    # if len(node) == 0:
    if not isinstance(node, dict):
        # print(node)
        return
    # for i in range(level):
    #     print("\t", )
    # print("traversal----------------")
    #print((typenode))
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
        new_tree = val[value]
        result = inference(data, attributes, new_tree)
    return result

def main():
    dataset = []
    dataset,label_list = load_data(filename)
    # print(dataset)
    attributes = dataset[0]
    # attributes.remove('id')
    # attributes.remove("Label")

    dataset.remove(attributes)
    # dataset.
    # print(dataset[0:2])
    gain0 = gain(attributes, dataset, "Customer", "Label")
    # gain_test = gain(attributes, dataset, "Outlook", "Play Golf")

    # Run ID3 to build
    # print(gain0)
    # print(gain_test)
    dtree = build_decision_tree(dataset, attributes, "Label", 0)
    print("Start to generate decision tree...")

    print(dataset[0:2])
    print("dtree")
    print(dtree)
    print("dtree")
    print(dtree.keys())

    traversal(dtree, 0)
    prediction = []
    for data in dataset:
        prediction.append(inference(data, attributes, dtree))

    count = 0
    for i in range(len(prediction)):
        if prediction[i] == label_list[i]:
            count += 1
    print(count/len(prediction))
if __name__ == '__main__':
    main()

# print(type(dataset))
# print(dataset)
# exit(0)
#
# X = dataset[1:]
# X = X[:, 1:8]
# X_res = numpy.array(X).astype("float")
# print(X_res)
