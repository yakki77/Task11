import csv
import numpy
import codecs
import math


## Load a csv file
filename = "csv_result-trainProdIntro-binary.csv"

# test dataset
# filename = "test.csv"

## Load dataset
def load_data(filename):
    dataset = []
    with codecs.open(filename, "r") as csvfile:
        csvReader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        # dataset = list(csvReader)
        for row in csvReader:
            # print(row)
            dataset.append(row[0].split(','))
        # dataset.append(row)
        # print(dataset)
        return dataset
        # print(', '.join(row))

# Calculate entropy E(X)
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


## Gain(T, X) = Entropy(T) - Entropy(T, X)
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

    ## Calculate the sum
    ## E(T, X) = Sum P(a) * E(a)
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


## get values in the attribute, such as "Student", "Business", "Professional"
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


def most_common_val(attributes, dataset, label):
    attribute_val_freq_map = {}
    index = 0
    for index in range(0, len(attributes)):
        if label == attributes[index]:
            break
    print("index for build_decision_tree attribute is " + str(index))

    ## Sum the frequency of each of the values in target attributes
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
    ## Build a new decision tree based on the examples given
    updated_dataset = dataset[:]
    index = 0
    for index in range(0, len(attributes)):
        if label == attributes[index]:
            break
    print("index for build_decision_tree attribute is " + str(index))
    ## Store the label_vals into a label_values list
    label_vals = []
    for row in dataset:
        label_val = row[index]
        label_vals.append(label_val)
    default = most_common_val(attributes, dataset, label)
    ## Terminate conditions
    if not dataset or (len(attributes)) <= 0:
        return default
    elif label_vals.count(label_vals[0]) == len(label_vals):
        return label_vals[0]
    else:
        ## Select the best next attributes that can optimal classify dataset
        best_attr = select_best_attribute(dataset, attributes, label)

        ## Build a new decision tree with best attribute and
        decision_tree = {best_attr: {}}
        values = get_values(dataset, attributes, best_attr)
        for val in values:
            new_dataset = get_newDataset(dataset, attributes, best_attr, val)
            newAttr = attributes[:]
            newAttr.remove(best_attr)
            subtree = build_decision_tree(new_dataset, newAttr, label, recurion)
            # Add the new subtree to the empty dictionary
            decision_tree[best_attr][val] = subtree
    return decision_tree


def traversal(node, level):
    if len(node) == 0:
        return
    for i in range(level):
        print("\t", )
    print("traversal")
    print(node)
    for val in node.keys():
        traversal(node[val], level + 1)


def main():
    dataset = []
    dataset = load_data(filename)
    # print(dataset)
    attributes = dataset[0]
    # attributes.remove('id')
    attributes.remove("Label")

    dataset.remove(attributes)
    # print(dataset[0:2])

    gain0 = gain(attributes, dataset, "Customer", "Label")

    ## For testing
    # gain_test = gain(attributes, dataset, "Outlook", "Play Golf")

    ## Run ID3 to build
    # print(gain0)
    # print(gain_test)
    dtree = build_decision_tree(dataset, attributes, "Label", 0)
    print("Start to generate decision tree...")

    # print(dataset[0:2])
    print("dtree")
    # print(dtree)
    # print("dtree")

    traversal(dtree, 0)


if __name__ == '__main__':
    main()
