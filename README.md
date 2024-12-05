## CSC 5601: Theory of Machine Learning (Graduate) - Isolation Forest (Anomaly Detection)


### Project Approach:
- Data Prep: Input a dataset with $n$ samples and $d$ features. Normalize + handle missing values/remove duplicates
- Implement Isolation Forest from scratch - building and interpreting isolation trees.
    - Add complexity analysis and optimization decisions in the implementation
- Compare with `sklearn.ensemble.IsolationForest`, discuss how random partitioning impacts runtime and prediction speed.
- Evaluate with performance metrics like the F1-score and AUC-ROC.


### Steps taken to build Isolation Forest algorithm from scratch, based on original paper

**Step 1: Build a single isolation tree**:

For each Isolation Tree:
1. Base Case: 
    - Stop splitting when:
        - The tree reaches a predefined height limit, $h_{max}$, based on $log_{2}(n)$
        - A node contains a single point
2. Recursive Splitting:
    - Randomly select a feature $f$ from the $d$ features.
    - Choose a random split value s within the range of the selected feature f's values.
    - Divide the data into 2 subsets:
        - $X_{left} = {{x ∈X∣x[f]<s}}$
        - $X_{right} = {{x ∈X∣x[f]≥s}}$
    - Create child nodes for $X_{left}$ and $X_{right}$, and repeat the process recursively.
3. Depth Tracking:
    - Record the depth at which each sample gets isolated. Anomalies are expected to be isolated at shallower depths.

**Step 4: Build the Forest**
- Create $T$ Isolation Trees by repeating the process above.
- Use random subsets of the data ($n_{subsample}$) for building each tree to improve robustness and efficiency.


**Step 5: Calculate Anomaly Scores**

For each data point $x$:
1.  Average Path Length: 
    - Compute the average path length, $E[h(x)]$, over all trees, where $h(x)$ is the depth at which $x$ is isolated.
2. Normalization: 
    - Normalize the path length using $c(n)$, the average path length of an unsuccessful search in a binary tree where $H(i)$ is the harmonic number:
$$c(n)=2H(n−1) − \frac{2(n−1)}{n}$$
3. Anomaly Score: 
    - Calculate the anomaly score $s(x,n)$:
    $$ s(x,n)= 2^{- \frac{E[h(x)]}{c(n)}}$$
    - Scores close to 1 indicate anomalies, while scores close to 0 suggest normality.


**Step 6: Threshold for Anomaly Detection**

Decide a threshold $τ$ for anomaly scores to classify data points as anomalies or normal instances.


**Step 7: Optimization Additions**

- improve representation for imbalanced datasets
    - done with adding `stratify` parameter in `IsolationForest` implementaion: By passing `stratify_labels`, the `stratified_sample` method will make sure that each subsample contains approximately the same proportion of fraud and non-fraud cases as the entire dataset.
- better split decisions for better tree balance
- better efficiency with feature subsampling
    - if there are many features, it doesn't make sense to evaluate all of them at every node
    - add `max_features` paramter to restrict the selection to a limited number of features
- TODO: add details

### Sources:
- https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf?q=isolation-forest