import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Node():
    def __init__(self, feature=None, threshold=None, left_node=None, right_node=None, information_gain=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left_node = left_node
        self.right_node = right_node
        self.information_gain = information_gain
        self.value = value

class DecisionTree():
    def __init__(self, min_samples=2, max_depth=4):
        self.root = None
        self.min_samples = min_samples
        self.max_depth = max_depth

    def buildTree(self, dataframe, current_depth=0):
        # splitting features and targets 
        X = dataframe[:, :-1] 
        Y = dataframe[:, -1]  
        # dimensions of dataframe
        samples, features = np.shape(X)

        # stopping conditions for not splitting the tree farther
        if current_depth <= self.max_depth and samples >= self.min_samples:
            # get_node_split returns the best split of the current node
            print("node split called")
            splitter = self.get_node_split(dataframe, samples, features)
            print("splitter:", splitter)
            if splitter['information_gain'] > 0:
                # recursion to build the subtrees of the childeren nodes
                left_child_subtree = self.buildTree(splitter['dataframe_left'], current_depth + 1)
                right_child_subtree = self.buildTree(splitter['dataframe_right'], current_depth + 1)
                # returns the decision node
                return Node(splitter['feature'], splitter['threshold'], left_child_subtree, right_child_subtree, splitter['information_gain'])

        # if conditions are not met, then then the leaf value is returned from the Node class
        leaf_node_value = self.calc_leaf_node(Y)
        return Node(value = leaf_node_value)

    def get_node_split(self, dataframe, samples, features):
        splitter = {} # stores the best split for the current node
        info_gain_max = ~sys.maxsize

        # loop which hits all features
        for feature in range(features):
            feature_val = dataframe[:, feature]
            all_thresholds = np.unique(feature_val) # returns all the unique values of a particular feature in the dataset
            
            # loops through all features only present in the dataset
            for threshold in all_thresholds:
                # split fuction returns the split of the left and right child nodes
                print("split called")
                dataframe_left, dataframe_right = self.split(dataframe, feature, threshold)
                # if childeren nodes are not 0
                if len(dataframe_left) > 0 and len(dataframe_right) > 0:
                    y = dataframe[:, -1]
                    left_y = dataframe_left[:, -1]
                    right_y = dataframe_right[:, -1] 
                    # calc_info_gain returns the information gain value (uses gini indexing)
                    info_gain_count = self.calc_info_gain(y, left_y, right_y, "gini")
                    # check for the current info gain is greater than the max info gain
                    if info_gain_count > info_gain_max:
                        splitter['feature'] = feature
                        splitter['threshold'] = threshold
                        splitter['dataframe_left'] = dataframe_left
                        splitter['dataframe_right'] = dataframe_right
                        splitter['information_gain'] = info_gain_count
                        info_gain_max = info_gain_count

        return splitter

    def split(self, dataframe, feature, threshold):
        # splits into left and right childeren
        dataframe_left = np.array([row for row in dataframe if row[feature] <= threshold]) # meets threshold condition
        dataframe_right = np.array([row for row in dataframe if row[feature] > threshold]) # greater than threshold
        return dataframe_left, dataframe_right

    def calc_info_gain(self, parent_root, left_child_node, right_child_node, mode="entropy"):
        # relative sizes of child nodes with respect to parent nodes
        left_node_weight = len(left_child_node) / len(parent_root)
        right_node_weight = len(right_child_node) / len(parent_root)

        # gini index = 1 - summation( [Pi]^2 )
            # Pi = probability of class i
        # gini index is used to save computation time due to info gain using logarithm
        if mode == "gini":
            info_gain = self.calc_gini(parent_root) - ( (left_node_weight * self.calc_gini(left_child_node)) + (right_node_weight * self.calc_gini(right_child_node)) )
        else:
            info_gain = self.calc_entropy(parent_root) - ( (left_node_weight * self.calc_entropy(left_child_node)) + (right_node_weight * self.entropy(right_child_node)) )

        return info_gain

    def calc_gini(self, y):
        gini_index = 0
        n_label = np.unique(y)

        for i in n_label:
            p = len(y[y == i]) / len(y)
            gini_index += p**2
            
        return 1 - gini_index

    def calc_entropy(self, y):
        entropy = 0
        n_label = np.unique(y)
    
        for i in n_label:
            p = len(y[y == i]) / len(y)
            entropy += -p * np.log2(p)

        return entropy

    def calc_leaf_node(self, Y):
        Y = list(Y)
        return max(Y, key=Y.count)

    def decision_tree_visualization(self, tree=None, indent=" "):
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)
        else:
            print("X " + str(tree.feature), "<=", tree.threshold, "?", tree.information_gain)
            print("%sleft_node:" % (indent), end="")
            self.decision_tree_visualization(tree.left_node, indent + indent)
            print("%sright_node:" % (indent), end="")
            self.decision_tree_visualization(tree.right_node, indent + indent)

    def fitter(self, X, Y):
        dataframe = np.concatenate((X, Y), axis=1)
        self.root = self.buildTree(dataframe)

    def predict(self, X):
        predictions = [self.prediction(x, self.root) for x in X]
        return predictions

    def prediction(self, x, tree):
        if tree.value != None: 
            return tree.value
        feature_num = x[tree.feature]
        if feature_num <= tree.threshold:
            return self.prediction(x, tree.left_node)
        else:
            return self.prediction(x, tree.right_node)


dataframe = pd.read_csv("./../preprocessing/data_bin.csv")

X = dataframe.iloc[:, :-1].values
print("X output")
print(X)
Y = dataframe.iloc[:, -1].values.reshape(-1,1)
print("Y output")
print(Y)
training_setX, testing_sample_setX, training_setY, testing_sample_setY = train_test_split(X, Y, test_size = .2, random_state=50)

classify = DecisionTree(min_samples=3, max_depth=3)
classify.fitter(training_setX, training_setY)
classify.decision_tree_visualization()

predictionY = classify.predict(testing_sample_setX) 
accuracy = accuracy_score(testing_sample_setY, predictionY)
print("Accuracy:", accuracy)