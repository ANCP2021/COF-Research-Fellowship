import numpy as np
import pandas as pd

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
        X = dataframe[:, :-1]
        print(X)
        Y = dataframe[:, -1]
        print(Y)


dataframe = pd.read_csv("./test.csv")
print(dataframe)
print("*********")
DecisionTree.buildTree(dataframe, 0)
