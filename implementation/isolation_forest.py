import numpy as np
import math
import scipy as sc
from isolation_tree import IsolationTree

class IsolationForest:
    """
    implements Algorithm 1: iForest from original Isolation Forest paper
    """

    def __init__(self, t, sub_sampling_size):
        self.t = t # number of trees
        self.sub_sampling_size = sub_sampling_size # sub sampling size ψ
        self.forest = None


    def fit(self, X:np.array):
        # initialize forest - stores isolation trees
        self.forest = []
        # set height limit using paper's formula
        self.height_limit = math.ceil(math.log2(self.sub_sampling_size))
        # repeat t times:
        for i in range(self.t):
            # X' ← sample(X, ψ) [randomly select subset of ψ samples from dataset X (subset denoted as X')
            if self.sub_sampling_size < X.shape[0]:
                X_subsample = X[np.random.choice(X.shape[0], self.subsample_size, replace=False)]
            else:
                X_subsample = X  # if subsample size > data size, use all data
            # Forest ← Forest ∪ iTree(X', 0, l)
            # Build an IsolationTree with X' as the inputdata, starting height = 0, and height limit = l
            # then add the IsolationTree to the forest
            tree = IsolationTree(e=0, limit=self.limit).fit(X_subsample)
            self.forest.append(tree)
        # output is forest (in self.forest list)    
        return self