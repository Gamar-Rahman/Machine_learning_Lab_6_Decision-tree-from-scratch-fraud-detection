import numpy as np
np.random.seed(1)

#Problem 1

def gini_score(groups, classes):
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    n_instances = float(sum(len(group) for group in groups))
    gini = 0.0

    for group in groups:
        size = float(len(group))
        if size == 0:
            continue

        score = 0.0
        labels = [row[-1] for row in group]

        for class_val in classes:
            p = labels.count(class_val) / size
            score += p * p

        gini += (1.0 - score) * (size / n_instances)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return gini

def create_split(index, threshold, datalist):
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    left, right = [], []

    for row in datalist:
        if float(row[index]) < float(threshold):
            left.append(row)
        else:
            right.append(row)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return left, right


def get_best_split(datalist):
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    class_values = list(set(row[-1] for row in datalist))
    best_index, best_value, best_score, best_groups = None, None, float("inf"), None

    for index in range(len(datalist[0]) - 1):
        for row in datalist:
            groups = create_split(index, row[index], datalist)
            gini = gini_score(groups, class_values)

            if gini < best_score:
                best_index = index
                best_value = row[index]
                best_score = gini
                best_groups = groups

    node = {'index': best_index, 'value': best_value, 'groups': best_groups}
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return node


def to_terminal(group):
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    labels = [row[-1] for row in group]
    label = max(set(labels), key=labels.count)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return label

def recursive_split(node, max_depth, min_size, depth):
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    left, right = node['groups']
    del(node['groups'])

    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return node

    if depth >= max_depth:
        node['left'] = to_terminal(left)
        node['right'] = to_terminal(right)
        return node

    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_best_split(left)
        recursive_split(node['left'], max_depth, min_size, depth + 1)

    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_best_split(right)
        recursive_split(node['right'], max_depth, min_size, depth + 1)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return node


def build_tree(train, max_depth, min_size):
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    root = get_best_split(train)
    recursive_split(root, max_depth, min_size, 1)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return root

def predict(root, sample):
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    if float(sample[root['index']]) < float(root['value']):
        if isinstance(root['left'], dict):
            return predict(root['left'], sample)
        else:
            return root['left']
    else:
        if isinstance(root['right'], dict):
            return predict(root['right'], sample)
        else:
            return root['right']
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


def Problem2():
    '''
    Inputs: 
    Outputs:
    acc: accuracy score, a real number
    f1: f1 score, a real number
    '''
    import pandas as pd
    import urllib.request
    import shutil
    from csv import reader
    from random import seed

    url = 'https://www.cs.uic.edu/~zhangx/teaching/data_banknote_authentication.csv'
    file_name = 'data_banknote_authentication.csv'
    with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)

    file = open(file_name, "rt")
    lines = reader(file)

    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    seed(1)

    dataset = list(lines)
    max_depth = 6
    min_size = 10
    num_train = 1000

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    train = dataset[:num_train]
    test = dataset[num_train:]

    tree = build_tree(train, max_depth, min_size)

    predictions = []
    actual = []

    for row in test:
        predictions.append(predict(tree, row))
        actual.append(row[-1])

    acc = accuracy_score(actual, predictions)
    f1 = f1_score(actual, predictions, pos_label='1')

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return acc, f1