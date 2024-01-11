import pandas as pd
import numpy as np
import random

class my_KMeans:

    def __init__(self, n_clusters=8, init = "k-means++", n_init = 10, max_iter=300, tol=1e-4):
        # init = {"k-means++", "random"}
        # use euclidean distance for inertia calculation.
        # stop when either # iteration is greater than max_iter or the delta of self.inertia_ is smaller than tol.
        # repeat n_init times and keep the best run (cluster_centers_, inertia_) with the lowest inertia_.
        self.n_clusters = int(n_clusters)
        self.init = init
        self.n_init = n_init
        self.max_iter = int(max_iter)
        self.tol = tol
        self.classes_ = range(n_clusters)
        # Centroids
        self.cluster_centers_ = None
        # Sum of squared distances of samples to their closest cluster center.
        self.inertia_ = None
    
    #included change
    def dist(self, a, b):
        # Compute Euclidean distance between a and b
        return np.sum((np.array(a)-np.array(b))**2)**(0.5)

    def initiate(self, X):
        # Initiate cluster centers
        # Input X is numpy.array
        # Output cluster_centers (list)

        # choose n_clusters random data points from the input data X as the initial cluster centers.
        if self.init == "random":
            # Select in a random fashion, n_clusters unique indices from the range of data points
            random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
            # Indices are used to extract the corresponding data points from X, which become the initial cluster centers.
            cluster_centers = X[random_indices]


        elif self.init == "k-means++":
            # Randomly initialize the first cluster center
            cluster_centers = [X[np.random.choice(X.shape[0])]] 
            # Iteratively selects the remaining cluster centers 
            while len(cluster_centers) < self.n_clusters:
                distances = np.array([min([np.linalg.norm(x-c)**2 for c in cluster_centers])for x in X])
                probabilities = distances / distances.sum()
                next_center = X[np.random.choice(X.shape[0], p=probabilities)]
                cluster_centers.append(next_center)
        else:
            raise Exception("Unknown value of self.init.")
        return cluster_centers

    def fit(self, X):
        # X: pd.DataFrame, independent variables, float        
        # repeat self.n_init times and keep the best run 
            # (self.cluster_centers_, self.inertia_) with the lowest self.inertia_.
        # write your code below
        X_feature = X.to_numpy()
        for i in range(self.n_init):
            cluster_centers, inertia = self.fit_once(X_feature)
            if self.inertia_ == None or inertia < self.inertia_:
                self.inertia_ = inertia
                self.cluster_centers_ = cluster_centers
        return
    
    def fit_once(self, X):
    # Fit K-means algorithm once
        cluster_centers = self.initiate(X)
        last_inertia = None

        for i in range(self.max_iter + 1):
            # Calculate pairwise distances between data points and cluster centers
            distances = np.array([np.linalg.norm(X - center, axis=1) for center in cluster_centers]).T

            # Assign data points to the nearest cluster
            labels = np.argmin(distances, axis=1)
            
            # Calculate the new cluster centers
            cluster_centers_new = np.array([X[labels == j].mean(axis=0) for j in range(self.n_clusters)])
            
            # Calculate inertia (sum of squared distances to the nearest cluster center)
            inertia = np.sum(np.min(distances, axis=1)**2)
            
            # Check for convergence
            if last_inertia is not None and last_inertia - inertia < self.tol:
                break

            # Update cluster centers
            cluster_centers = cluster_centers_new
            last_inertia = inertia
        return cluster_centers, inertia


    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        # write your code below
        predictions = [np.argmin(dist) for dist in self.transform(X)]
        return predictions
    
    def transform(self, X):
        # Transform to cluster-distance space
        # X: pd.DataFrame, independent variables, float
        # return dists = list of [dist to centroid 1, dist to centroid 2, ...]
        # write your code below
        dists = [[self.dist(x, centroid) for centroid in self.cluster_centers_] for x in X.to_numpy()]
        return dists

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)





