import numpy as np
import scipy as sc

class IsolationTree:
    # implements Algorithm 2: iTree from original Isolation Forest paper

    def __init__(self, e, limit):
        self.e = e # current tree height
        self.limit = limit # height limit
        self.right_subtree = None
        self.left_subtree = None
        self.split_point_p = None
        self.attribute_q = None
        self.size = None

    def fit(self, X:np.array):

        # base case - stops splitting when at heigh limit or node contains single point
        # return the leaf node
        if self.e >= self.l or X.shape[0] <=1:
            self.size = X.shape[0] # holds the size of subset X 
            return self # (this is the external node)
        else:
            # Q is list of attributes in X (data)
            # choose random attribute q i.e. feature from data
            self.attribute_q = self.randomly_select_attribute(X)
            # pick random split value p within range of value for attribute
            attribute_q_data = X[:,self.attribute_q]
            self.split_point_p = self.randomly_select_split_point(attribute_q_data)
            # split data into left and right based on the split point
            # X_l ← filter(X, q < p)
            X_left = X[attribute_q_data < self.split_point_p]
            # X_r ← filter(X, q ≥ p)
            X_right = X[attribute_q_data >= self.split_point_p]
            # recursive splitting
            # build left subtree increasing current height to e + 1
            self.left_subtree = IsolationTree(self.e + 1, self.limit).fit(X_left)
            # also build right subtree
            self.right_subtree = IsolationTree(self.e + 1, self.limit).fit(X_right)
            # store random split attribute q and split value p in an internal Node
            #TODO?
            # output: an iTree
            return self

    def randomly_select_attribute(self, X):
        # X is data
        # randomly select an attribute (column index) from data
        return np.random.randint(0, X.shape[1])

    def randomly_select_split_point(self, attribute_q_data):
        # randomly select a split point p from max and min values of attribute q in X
        # attribute_q_data -> 1D array of values for randomly selected attribute q
        # retruns a randomly selected split point
        min_value = attribute_q_data.min()
        max_value = attribute_q_data.max()
        split_point = np.random.uniform(min_value, max_value)
        return split_point