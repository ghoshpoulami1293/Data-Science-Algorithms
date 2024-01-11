import pandas as pd
import numpy as np
from collections import Counter

class my_NB:

    #no change included
    def __init__(self, alpha=1):
        # alpha: smoothing factor
        # P(xi = t | y = c) = (N(t,c) + alpha) / (N(c) + n(i)*alpha)
        # where n(i) is the number of available categories (values) of feature i
        # Setting alpha = 1 is called Laplace smoothing
        self.alpha = alpha


    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, str
        # y: list, np.array or pd.Series, dependent variables, int or str

        # list of classes for this model
        self.classes_ = list(set(list(y)))

        # for calculation of P(y)
        self.P_y = Counter(y)

        # self.P[yj][Xi][xi] = P(xi|yj) where Xi is the feature name and xi is the feature value, yj is a specific class label        
        # Calculate P(yj) and P(xi|yj)        
        # make sure to use self.alpha in the __init__() function as the smoothing factor when calculating P(xi|yj)
        # write your code below

        # initializes an empty dictionary called self.P to store conditional probabilities.
        self.P = {}

        #Iterate over each class label in self.classes_
        for classlabel in self.classes_:
            # Initialize a sub-dictionary for each class label to store conditional probabilities for each feature.
            self.P[classlabel] = {}  
            #iterates over each feature 
            for eachfeature in X.columns:
                 # Initialize a sub-dictionary for each feature
                self.P[classlabel][eachfeature] = {} 
                # Calculate P(xi|yj) for each each unique value of the feature.
                for eachval in X[eachfeature].unique():
                    # Count of the number of occurrences where the feature xi takes the specific value value within the class yj
                    countOf_xi_yj = len(X[(X[eachfeature] == eachval) & (y == classlabel)])
                    # Count of the number of occurrences of class yj
                    countOf_yj = len(y[y == classlabel])
                    # Calculate the number of unique values for the feature xi
                    numberOfUniqueVal_xi = len(X[eachfeature].unique())
                    
                    # count of occurrences of the feature value xi within the class yj
                    countOfOccur_xi_pj = (countOf_xi_yj + self.alpha)
                    # count of occurrences of the class yj plus a smoothing factor (Laplace smoothing) 
                    countOfOccur_yj_Lap  = (countOf_yj + numberOfUniqueVal_xi * self.alpha)
                    # Calculate the probability P(xi|yj) using Laplace smoothing:
                    prob_xi_yj = countOfOccur_xi_pj / countOfOccur_yj_Lap

                    # Store P(xi|yj) in self.P dictionary
                    self.P[classlabel][eachfeature][eachval] = prob_xi_yj
        return
    
    def predict(self, X):
        # X: pd.DataFrame, independent variables, str
        # return predictions: list
        # write your code below        

        # initialize list to store predicted class labels for each data point in X
        predictions = []

        #iterate through each row of the dataframe
        for _, row in X.iterrows():
            #initialize values 
            maximum_prob = float('-inf')
            predicted_class = None
            # iterate over each class label in self.classes_
            for eachlabel in self.classes_:
                #initialize the probability of each class
                probability = 1.0 
                #iterates over each feature and its corresponding value in the particular row.
                for eachfeature, eachvalue in row.iteritems():
                    #calculate the probability for each feature-value pair :
                    #if(feature value is not found in the self.P), calculate probability.
                    #else use Laplace smoothing to calculate the probability
                    if eachvalue in self.P[eachlabel][eachfeature]:
                        probability *= self.P[eachlabel][eachfeature][eachvalue]
                    else:
                        lapSmoothFactor = self.alpha  
                        probDen = (len(X[X.index == eachlabel]) + len(X[eachfeature].unique()) * self.alpha)                      
                        probability *= lapSmoothFactor / probDen
                #Calculate the final probability for all the features
                probability *= (self.P_y[eachlabel] / sum(self.P_y.values()))
                # update predicted_class to current class label if current class label> maximum probability
                if probability > maximum_prob:
                    maximum_prob = probability
                    predicted_class = eachlabel
            predictions.append(predicted_class)

        return predictions

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, str
        # prob is a dict of prediction probabilities belonging to each categories
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)                
        # P(yj|x) = P(x|yj)P(yj)/P(x)
        # write your code below

        #initialize dictionary to store the predicted probabilities for each class label
        probs = {}
        #iterate over each class label in self.classes_
        for eachlabel in self.classes_:
            #initialize probability for the current class label
            probability = self.P_y[eachlabel]
            #iterate over each feature to calculate conditional probability
            for key in X:
                valueOfColumnn=X[key]
                probability *= valueOfColumnn.apply(lambda value: self.P[eachlabel][key][value] if value in self.P[eachlabel][key] else 1)
            probs[eachlabel] = probability
        probs = pd.DataFrame(probs, columns=self.classes_)
        # calculate sum of probabilities across each row 
        sums = probs.sum(axis=1)
        #normalize the probabilities in each row
        probs = probs.apply(lambda v: v / sums)
        return probs



