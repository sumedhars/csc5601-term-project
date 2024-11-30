import numpy as np
import math
import scipy as sc
from isolation_tree import IsolationTree

class IsolationForest:
    """
    implements Algorithm 1: iForest from original Isolation Forest paper
    """

    def __init__(self, t, sub_sampling_size, max_features=None, stratify=None):
        """
        :param t: number of trees
        :param sub_sampling_size: subsampling size ψ
        :param max_features: max number of features to consider at each split - optimization
        :param stratify: array of labels for stratified sampling (optional) - optimization
        """
        self.t = t 
        self.max_features = max_features
        self.stratify = stratify  
        self.sub_sampling_size = sub_sampling_size 
        self.forest = None


    def stratified_sample(self, X):
        """
        optimization addition (not in paper):
        do stratified sampling to ensure proportional representation of data based on labels
        """
        unique_labels, counts = np.unique(self.stratify, return_counts=True)
        proportions = counts / counts.sum()

        stratified_indices = []
        for label, proportion in zip(unique_labels, proportions):
            label_indices = np.where(self.stratify == label)[0]
            sample_size = int(round(self.sub_sampling_size * proportion))
            sampled_indices = np.random.choice(label_indices, sample_size, replace=False)
            stratified_indices.extend(sampled_indices)

        return X[np.array(stratified_indices)]


    def fit(self, X:np.array):
        # initialize forest - stores isolation trees
        self.forest = []
        # set height limit using paper's formula
        self.height_limit = math.ceil(math.log2(self.sub_sampling_size))
        # repeat t times:
        for i in range(self.t):
            # stratified sampling if labels are provided
            if self.stratify is not None:
                X_subsample = self.stratified_sample(X)
            else:
                # default sampling
                # X' ← sample(X, ψ) [randomly select subset of ψ samples from dataset X (subset denoted as X')
                if self.sub_sampling_size < X.shape[0]:
                    X_subsample = X[np.random.choice(X.shape[0], self.sub_sampling_size, replace=False)]
                else:
                    X_subsample = X  # if subsample size > data size, use all data
            # Forest ← Forest ∪ iTree(X', 0, l)
            # Build an IsolationTree with X' as the inputdata, starting height = 0, and height limit = l
            # then add the IsolationTree to the forest
            tree = IsolationTree(e=0, limit=self.height_limit, max_features=self.max_features).fit(X_subsample)
            self.forest.append(tree)
        # output is forest (in self.forest list)    
        return self
    

    def anomaly_score(self, x):
        """
        Implements Equation 2 from the paper:
        In the evaluating stage, an anomaly score s is derived from the expected path length E(h(x)) 
        for each test instance. 
        When h(x) is obtained for each tree of the ensemble, an anomaly score is produced 
        by computing s(x, ψ) in Equation 2 (!!)
        """
        avg_path_length = self.average_path_length(x)
        c_subsampling = self.c_func(self.sub_sampling_size)
        anomaly_score = 2 ** (-avg_path_length/c_subsampling)
        return anomaly_score


    def average_path_length(self, x):
        """
        from the paper:
        E(h(x)) are derived by passing instances through each iTree in an iForest
        E(h(x)) is the average of h(x) from a collection of isolation trees.
        """
        total_path_length = sum(tree.path_length(x, 0) for tree in self.forest)
        return total_path_length / len(self.forest)


    def c_func(self, size):
        """
        implements Equation 1 from the paper:
        Given a data set of n instances, the average path length of unsuccessful search in BST
        """
        if size <= 1:
            return 0
        else:
            c = (2 * (np.log(size - 1) + np.euler_gamma)) - (2 * (size - 1) / size)
            return c