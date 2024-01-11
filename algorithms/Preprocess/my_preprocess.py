import numpy as np
from scipy.linalg import svd
from copy import deepcopy
from collections import Counter

class my_normalizer:
    def __init__(self, norm="Min-Max", axis = 1):
        #     norm = {"L1", "L2", "Min-Max", "Standard_Score"}
        #     axis = 0: normalize rows
        #     axis = 1: normalize columns
        self.norm = norm
        self.axis = axis

    def fit(self, X):
        #     X: input matrix
        #     Calculate offsets and scalers which are used in transform()
        X_array  = np.asarray(X)

        # Write your own code below - change starts
        dimension1, dimension2 = X_array.shape
        self.offsets = []
        self.scalers = []        
        if self.axis == 1:
            for column in range(dimension2):
                offset, scaler = self.vector_normalization(X_array[:, column])
                self.offsets.append(offset)
                self.scalers.append(scaler)
        elif self.axis == 0:
            for row in range(dimension1):
                offset, scaler = self.vector_normalization(X_array[row])
                self.offsets.append(offset)
                self.scalers.append(scaler)
        else:
            raise Exception("Axis is invalid")
        #change ends

    def transform(self, X):
        # Transform X into X_norm
        X_norm = deepcopy(np.asarray(X))

        # Write your own code below - change starts 
        dimension1, dimension2 = X_norm.shape
        if self.axis == 1:
            for column in range(dimension2):
                X_norm[:, column] = (X_norm[:, column]-self.offsets[column])/(self.scalers[column])
        elif self.axis == 0:
            for row in range(dimension1):
                X_norm[row] = (X_norm[row]-self.offsets[row])/(self.scalers[row])
        else:
            raise Exception("Unknown axis.")
        #change ends 
        return X_norm

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)   
     
    def vector_normalization(self, x):
        # Calculate the offset and scaler for input vector x
        
        if self.norm == "Min-Max":
            # Write your own code below
            X_min = np.min(x)
            X_max = np.max(x)
            #X_MinMaxNorm = (x - X_min) / (X_max - X_min)
            calculatedoffset = X_min
            calculatedscaler = X_max - X_min

        elif self.norm == "L1":
            # Write your own code below
            l1_norm = np.linalg.norm(x, ord=1)
            calculatedoffset = 0
            calculatedscaler = l1_norm

        elif self.norm == "L2":
            # Write your own code below
            l2_norm = np.linalg.norm(x, ord=2)
            calculatedoffset = 0
            calculatedscaler = l2_norm

        elif self.norm == "Standard_Score":
            # Write your own code below
            calculatedMean = np.mean(x)
            calculatedStandardDeviation = np.std(x)
            calculatedoffset = calculatedMean
            calculatedscaler = calculatedStandardDeviation
        else:
            raise Exception("The normalization is invalid.")        
        return calculatedoffset, calculatedscaler
    
class my_pca:
    def __init__(self, n_components = 5):
        #     n_components: number of principal components to keep
        self.n_components = n_components

    def fit(self, X):
        #  Use svd to perform PCA on X
        #  Inputs:
        #     X: input matrix
        #  Calculates:
        #     self.principal_components: the top n_components principal_components
        #   Vh = transpose of x 
        U, s, Vh = svd(X)
        # Write your own code below
        self.principal_components = Vh[:self.n_components, :]

    def transform(self, X):
        # X_pca = X.dot(self.principal_components)
        X_array = np.asarray(X)
        # Write your own code below - change starts 
        X_pca = X_array.dot(self.principal_components.T)        
        #change ends
        return X_pca

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

def stratified_sampling(y, ratio, replace=True):
    #  Inputs:
    #     y: class labels
    #     0 < ratio < 1: len(sample) = len(y) * ratio
    #     replace = True: sample with replacement
    #     replace = False: sample without replacement
    #  Output:
    #     sample: indices of stratified sampled points
    #             (ratio is the same across each class,
    #             samples for each class = int(np.ceil(ratio * # data in each class)) )
    if ratio<=0 or ratio>=1:
        raise Exception("ratio must be 0 < ratio < 1.")    
    y_array = np.asarray(y)  

    # Write your own code below
    classes = np.unique(y_array)
    sample = []

    for label in classes:
        
        #Creating boolean array comparing elements of the y_array to the value label
        boolArray = y_array == label
        #Indices where the condition (equality to label) is True
        trueIndices = np.where(boolArray)
        #Extract the array of indices
        calculatedIndices = trueIndices[0]    
               
        # 0 < ratio < 1: len(sample) = len(y) * ratio
        countofSamples = ratio * len(calculatedIndices)
        #typecasting the number of samples counted
        numOfSamples = int(np.ceil(countofSamples))

        # stratified sampling
        if replace:
            # Sample with replacement
            indicesSampled = np.random.choice(calculatedIndices, size=numOfSamples, replace=True)
        else:
            # Sample without replacement
            if numOfSamples > len(calculatedIndices):   
                #not acceptable condition             
                raise Exception("Error")
            indicesSampled = np.random.choice(calculatedIndices, size=numOfSamples, replace=False)

        # adding sampled indices to the array 
        sample.extend(indicesSampled)
    return np.array(sample).astype(int)