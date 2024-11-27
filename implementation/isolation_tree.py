import numpy as np
import scipy as sc

class IsolationTree:
    # implements Algorithm 2: iTree from original paper

    def __init__(self, e, limit):
        self.e = e # current tree height
        self.limit = limit # height limit

    def fit(self, X:np.array):

        # base case - stops splitting when at heigh limit or node contains single point
        # return the leaf node
        if self.e >= self.l or X.shape[0] <=1:
            self.size = X.shape[0] # holds the size of subset X 
            return self # (this is the external node)
        else:
            # Q is list of attributes in X (data)
            # choose random attribute q i.e. feature from data
            self.random_attribute = self.randomly_select_attribute(X)
            # pick random split value p within range of value for attrivutes
            pass

    def randomly_select_attribute(self, X):
        # X is data
        # randomly select an attribute (column index) from data
        return np.random.randint(0, X.shape[1])

    def randomly_select_split_point(self):
        pass