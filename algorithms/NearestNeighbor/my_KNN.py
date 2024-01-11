import pandas as pd
import numpy as np
from collections import Counter

class my_KNN:

    def __init__(self, n_neighbors=5, metric="minkowski", p=2):
        # metric = {"minkowski", "euclidean", "manhattan", "cosine"}
        # p value only matters when metric = "minkowski"
        # notice that for "cosine", 1 is closest and -1 is furthest
        # therefore usually cosine_dist = 1- cosine(x,y)
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.p = p

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array or pd.Series, dependent variables, int or str
        self.classes_ = list(set(list(y)))
        # write your code below        
        self.X = X
        self.y = y
        return
    
    def dist(self, x):
            # Calculate distances of training data to a single input data point (distances from self.X to x)
            # Output np.array([distances to x])
            if self.metric == "minkowski":
                #write your own code
                distances = np.linalg.norm(self.X - x, ord=self.p, axis=1)
            elif self.metric == "euclidean":
                #write your own code
                distances = np.linalg.norm(self.X - x, ord=2, axis=1)
            elif self.metric == "manhattan":
                #write your own code
                distances = np.sum(np.abs(self.X - x), axis=1)
            elif self.metric == "cosine":
                #write your own code
                L2normValue_x = np.linalg.norm(x)
                L2normValue_X = np.linalg.norm(self.X, axis=1)
                cosineSimilarity = np.dot(self.X, x) / (L2normValue_X * L2normValue_x)
                distances = 1 - cosineSimilarity
            else:
                raise Exception("Unknown criterion.")
            return distances
    


    def k_neighbors(self, x):
        # Return the stats of the labels of k nearest neighbors to a single input data point (np.array)
        # Output: Counter(labels of the self.n_neighbors nearest neighbors) e.g. {"Class A":3, "Class B":2}
        distances = self.dist(x)

        #sort indices of training data points based on their distances from the input data point x
        sortIndices = np.argsort(distances)

        #select indices of self.n_neighbors' closest data points from the sorted list
        nearestIndiceSelection = sortIndices[:self.n_neighbors]

        #retrieve labels corresponding to the selected nearest neighbor indices from self.y        
        nearestLabelSelection = []
        for item in nearestIndiceSelection:
            nearestLabelSelection.append(self.y[item])

        #count occurrences of each unique label in the nearestLabelSelection list
        output = Counter(nearestLabelSelection)
        return output



    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        # write your code below
        probs = self.predict_proba(X)        
        predictions = []
        for prob in probs.to_numpy():
            predclassIndex = np.argmax(prob)
            predictedClass = self.classes_[predclassIndex]
            predictions.append(predictedClass)
        return predictions
    


    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, float
        # prob is a dict of prediction probabilities belonging to each categories
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        # write your code below        
        probs = []

        try:
            #check if columns in the input DataFrame X matches columns in the training data stored in self.X
            X_feature = X[self.X.columns]
        except:
            raise Exception("Input data mismatch.")
        
        #iterate over each data point x in the input DataFrame X_feature
        for each_x in X_feature.to_numpy():
            #find the nearest neighbours of x in the training data and obtain a dictionary 
            # The dictionary returned will contain class labels as keys and count of each class among the k nearest neighbours as values
            neighbours = self.k_neighbors(each_x)
            
            # Initialize probabilities for all classes to 0
            # Initializes an empty dictionary probability to store the class probabilities for x. 
            probability = {}
            for eachlabel in self.classes_:
                probability[eachlabel] = 0.0           
            totalNumberOfNeighbours = sum(neighbours.values())
            
            # calculate probability as the count of each class / total number of nearest neighbors.
            if (totalNumberOfNeighbours != 0):
                for label, countOfClass in neighbours.items():
                    probability[label] = countOfClass / totalNumberOfNeighbours
            
            probs.append(probability)

        #create Data frame using the probabilities and return 
        probs = pd.DataFrame(probs, columns=self.classes_)
        return probs