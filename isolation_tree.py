import numpy as np
import scipy.optimize as opt
from collections import Counter

class IsolationTree:
    """
    implements Algorithm 2: iTree from original Isolation Forest paper
    """

    def __init__(self, e, limit, max_features):
        self.e = e # current tree height
        self.limit = limit # height limit
        # max num. of features to consider - control num of candidate attributes sampled for splitting
        self.max_features = max_features 
        self.right_subtree = None
        self.left_subtree = None
        self.split_point_p = None
        self.attribute_q = None
        self.size = None

    def fit(self, X:np.array):

        # base case - stops splitting when at heigh limit or node contains single point
        # return the leaf node
        if self.e >= self.limit or X.shape[0] <=1:
            self.size = X.shape[0] # holds the size of subset X 
            return self # (this is the external node)
        else:
            # Q is list of attributes in X (data)
            # choose random attribute q i.e. feature from data
            self.attribute_q = self.randomly_select_attribute_opt(X)
            # pick random split value p within range of value for attribute
            attribute_q_data = X[:,self.attribute_q]
            self.split_point_p = self.optimize_split_point(attribute_q_data)
            # split data into left and right based on the split point
            # X_l ← filter(X, q < p)
            # X_r ← filter(X, q ≥ p)
            mask = attribute_q_data < self.split_point_p
            X_left, X_right = X[mask], X[~mask]
            # recursive splitting
            # build left subtree increasing current height to e + 1
            self.left_subtree = IsolationTree(self.e + 1, self.limit, self.max_features).fit(X_left)
            # also build right subtree
            self.right_subtree = IsolationTree(self.e + 1, self.limit, self.max_features).fit(X_right)
            # store random split attribute q and split value p in an internal Node
            #TODO?
            # output: an iTree
            return self


    def randomly_select_attribute(self, X):
        # X is data
        # randomly select an attribute (column index) from data
        return np.random.randint(0, X.shape[1])
    

    def randomly_select_attribute_opt(self, X):
        """
        optimization addition (not in paper):
        - randomly selecting attributes can lead to suboptimal splits
        - this method randomly selects a SUBSET of features before choosing a split attribute instead
        - also reduces the computational cost while still maintaining random splits
        - balances complexity + model quality, especially for high-dimensional datasets
        """
        if self.max_features:
            # subsample features
            random_features = np.random.choice(X.shape[1], min(self.max_features, X.shape[1]), replace=False)
            return np.random.choice(random_features)
        else:
            # use all features
            return np.random.randint(0, X.shape[1])


    def randomly_select_split_point(self, attribute_q_data):
        """
        randomly select a split point p from max and min values of attribute q in X
        :param attribute_q_data: 1D array of values for randomly selected attribute q
        :return: a randomly selected split point
        """
        min_value = attribute_q_data.min()
        max_value = attribute_q_data.max()
        split_point = np.random.uniform(min_value, max_value)
        return split_point
    

    def optimize_split_point(self, attribute_q_data):
        """
        optimization addition (not in paper):
        - use optimization to find a split that minimizes variance between left and right splits
        - helps create more balanced trees!
        - minimizing variance makes data more evenly divided - this reduces tree depth and 
            also improves performance
        """
        min_value = attribute_q_data.min()
        max_value = attribute_q_data.max()

        # variance-based optimization function
        def split_variance(p):
            left_size = (attribute_q_data < p).sum()
            right_size = attribute_q_data.size - left_size
            return left_size * right_size  # Variance proxy: minimize imbalance

        # optimize split point 
        result = opt.minimize_scalar(split_variance, bounds=(min_value, max_value), method='bounded')
        return result.x
    

    def path_length(self, x, e):
        """
        Implements Algorithm 3 from the paper:
        Using PathLength function, a single path length h(x) is derived by counting the 
        number of edges e from the root node to a terminating node as instance x traverses through an iTree

        :param x: an instance to evaluate (1D array of feature values) - test data point
        :param e: - current path length; to be initialized to zero when first called
        :return: single path length h(x)
        """
        # T is an external node when both right subtree and left subtree are None
        if self.left_subtree == None and self.right_subtree == None:
            # e + c(T.size)
            return e + self.c_func(self.size)
        # a ← T.splitAtt
        a = self.attribute_q
        # x_a is o the value of the attribute (or feature) a of the instance x
        if x[a] < self.split_point_p:
            # PathLength(x, T.left, e + 1)
            return self.left_subtree.path_length(x, e + 1)
        else: 
            # PathLength(x, T.right, e + 1)
            return self.right_subtree.path_length(x, e + 1)


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