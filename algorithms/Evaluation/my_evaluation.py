import numpy as np
import pandas as pd
from collections import Counter

class my_evaluation:
    # Binary class or multi-class classification evaluation
    # Each data point can only belong to one class
    
    def __init__(self, predictions, actuals, pred_proba=None):
        # inputs:
        # predictions: list of predicted classes
        # actuals: list of ground truth
        # pred_proba: pd.DataFrame of prediction probability of belonging to each class
        self.predictions = np.array(predictions)
        self.actuals = np.array(actuals)
        self.pred_proba = pred_proba
        if type(self.pred_proba) == pd.DataFrame:
            self.classes_ = list(self.pred_proba.keys())
        else:
            self.classes_ = list(set(list(self.predictions) + list(self.actuals)))
        self.confusion_matrix = None        
    
    def confusion(self):
        # compute confusion matrix for each class in self.classes_
        # self.confusion = {self.classes_[i]: {"TP":tp, "TN": tn, "FP": fp, "FN": fn}}
        # no return variables
        # write your own code below:

        # create boolean array to indicate correct/ incorrect prediction by :
        # comparison of the predicted class labels with the corresponding element in the self.actuals array
        # if prediction matches actual label , True, else, False
        correctPredictions = self.predictions == self.actuals

        #calculate the accuracy of the model : number of correct predictions/total number of predictions
        self.acc = float(Counter(correctPredictions)[True])/len(correctPredictions) 

        #initiallizing the confusion matrix
        self.confusion_matrix = {}     
        
        # iterating over each class in each class      
        for eachclasslabel in self.classes_:
            #i. evaluate the conditions for true positive, true negative, false positive and false negative.
            truePositive = (self.predictions == eachclasslabel) & (self.actuals == eachclasslabel)
            falsePositive = (self.predictions == eachclasslabel) & (self.actuals != eachclasslabel)
            falseNegative = (self.predictions != eachclasslabel) & (self.actuals == eachclasslabel)
            trueNegative = (self.predictions != eachclasslabel) & (self.actuals != eachclasslabel)

            # ii.calculate the number of elements of : true positive, true negative, false positive and false negative. 
            tp = np.sum(truePositive)           
            fp = np.sum(falsePositive) 
            fn = np.sum(falseNegative)           
            tn = np.sum(trueNegative)

            # values are stored in confusion matrix with keys "TP," "TN," "FP," and "FN" associated with the current class label.
            self.confusion_matrix[eachclasslabel] = {"TP":tp, "TN": tn, "FP": fp, "FN": fn}
        return 
        
    def accuracy(self):
        if self.confusion_matrix==None:
            self.confusion()              
        return self.acc

    def precision(self, target=None, average = "macro"):
        # compute precision
        # target: target class (str). If not None, then return precision of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average precision
        # output: prec = float
        # note: be careful for divided by 0
        # write your own code below:

        #if the confusion_matrix is of None type , call confusion() method to compute the confusion_matrix.
        if self.confusion_matrix==None:
            self.confusion()  

        # If target is present in the class and not None        
        if target in self.classes_:
            truepositive = self.confusion_matrix[target]["TP"]
            falsepositive = self.confusion_matrix[target]["FP"]
            if truepositive+falsepositive == 0:
                prec = 0
            else:
                # Calculates the precision for a specific class 
                prec = float(truepositive) / (truepositive + falsepositive)
        #if target is of the None Type, calculates the average precision based on the average parameter
        else:
            #calculate precision as count of true positives divided by count of false positives
            if average == "micro":                
                prec = self.accuracy()
            else:
                prec = 0
                n = len(self.actuals)
                for label in self.classes_:
                    truepositive = self.confusion_matrix[label]["TP"]
                    falsepositive = self.confusion_matrix[label]["FP"]
                    if truepositive + falsepositive == 0:
                        precsionlabelOftargetClass = 0
                    else:
                        precsionlabelOftargetClass = float(truepositive) / (truepositive + falsepositive)
                    # calculates precision for each class, then calculates unweighted average of class precision
                    if average == "macro":
                        ratio = 1 / len(self.classes_)
                    # calculate weighted average of precisions
                    elif average == "weighted":
                        ratio = Counter(self.actuals)[label] / float(n)
                    else:
                        raise Exception("Error due to unknown average.")
                    prec += precsionlabelOftargetClass * ratio
        return prec

    def recall(self, target=None, average = "macro"):
        # compute recall
        # target: target class (str). If not None, then return recall of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average recall
        # output: recall = float
        # note: be careful for divided by 0
        # write your own code below

        #if the confusion_matrix is of None type , call confusion() method to compute the confusion_matrix.
        if self.confusion_matrix ==  None:
            self.confusion()

        #calculating recall for the target class 
        if target:
            #retrieve true positive and false negatives
            truepositive = self.confusion_matrix[target]["TP"]
            falsenegative = self.confusion_matrix[target]["FN"]
            #to avoid divided by 0 error:
            if truepositive + falsenegative == 0:
                recall = 0
            #Calculate recall
            else:
                recall = truepositive / (truepositive + falsenegative)
        #if target==None, return average recall
        else:
            #calculate micro recall
            if average == "micro":
                recall = self.accuracy()
            else:
                recall = 0
                n = len(self.actuals)
                #calculate recall for each class
                for eachClassLabel in self.classes_:
                    truepositive = self.confusion_matrix[eachClassLabel]["TP"]
                    falsenegative = self.confusion_matrix[eachClassLabel]["FN"]
                    if truepositive + falsenegative == 0:
                        recall_eachClassLabel = 0
                    else:
                        recall_eachClassLabel = truepositive / (truepositive + falsenegative)
                    # calculates macro  recall
                    if average == "macro":
                        ratio = 1 / len(self.classes_)
                     # calculates weighted  recall
                    elif average == "weighted":
                        ratio = np.sum(self.actuals == eachClassLabel) / n
                    else:
                        raise Exception("Error due to unknown average.")
                    recall += recall_eachClassLabel * ratio        
            
        return recall
    
    def f1(self, target=None, average = "macro"):
        # compute f1
        # target: target class (str). If not None, then return f1 of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average f1
        # output: f1 = float
        # note: be careful for divided by 0
        # write your own code below
        
        #if target is not None , calculate F1 score for the target class
        if target:
            precison = self.precision(target = target, average=average)
            recall = self.recall(target = target, average=average)
            if (precison + recall == 0):
                f1_score = 0
            else:
                f1_score = (2.0 * precison * recall / (precison + recall))
        #if target is None
        else:
            #"write your own code"
            # calculate micro f1 score
            if average == "micro":
                f1_score = self.accuracy()
            else:
                f1_score = 0
                #Calculate f1 score for each class
                for eachClassLabel in self.classes_:
                    precison = self.precision(target=eachClassLabel, average=average)
                    recall = self.recall(target=eachClassLabel, average=average)
                    if precison + recall == 0:
                        f1_eachClassLabel = 0
                    else:
                        f1_eachClassLabel = (2.0 * precison * recall / (precison + recall))
                    # calculate macro f1 score
                    if (average == "macro"):
                        ratio = (1 / len(self.classes_))
                    # calculate weighted f1 score
                    elif (average == "weighted"):
                        ratio = (np.sum(self.actuals == eachClassLabel) / len(self.actuals))
                    else:
                        raise Exception("Error due to unknown average.")
                    f1_score += f1_eachClassLabel * ratio
        return f1_score

    def auc(self, target):
        # compute AUC of ROC curve for the target class
        # return auc = float

        #if self.pred_proba is Nonetype, return None, as the AUC or ROC cannot be calculated without predicted probablities
        if type(self.pred_proba) == type(None):
            return None
        
        #if the self.pred_proba is Nonetype, calculate AUC for ROC curve
        else:
            # write your own code below   
            # if the target class is present      
            if (target in self.classes_):

                #sorting the predicting probability values for the target class in descending order
                descendingOrderProbabilities = np.argsort(self.pred_proba[target])[::-1]

                #sorting corresponding actual labels of the sorted predicted probability values
                desc_order_classlabels = self.actuals[descendingOrderProbabilities]

                truepositive = 0
                falsepositive = 0                
                truePositiveRate = 0
                falsePositiveRate = 0
                truePositiveRate_values=[0]
                falsePositiveRate_values=[0]
                auc_target = 0
                
                #iterate over the sorted class labels
                for i in range(len(desc_order_classlabels)):
                    #if sorted label matches the target class , increment true positive element else increment false positive element
                    if (self.actuals[descendingOrderProbabilities])[i] == target:
                        truepositive += 1
                    else:
                        falsepositive += 1

                    #Calculate TPR and FPR
                    truePositiveRate = truepositive / np.sum(self.actuals[descendingOrderProbabilities] == target)
                    falsePositiveRate = falsepositive / np.sum(self.actuals[descendingOrderProbabilities] != target)
                    #add it to the TPR and FPR list
                    truePositiveRate_values.append(truePositiveRate)
                    falsePositiveRate_values.append(falsePositiveRate)

                    #calculates the area under the ROC curve (AUC) for the target class using prev and current TPR and FPR values
                    FPR_difference= (falsePositiveRate_values[-1] - falsePositiveRate_values[-2])
                    TPR_summation= (truePositiveRate_values[-1] + truePositiveRate_values[-2]) 
                    auc_target += ((FPR_difference * TPR_summation) / 2)
           
            # if  data set is not binary , throw exception
            elif(len(self.classes_) != 2):
                raise Exception("AUC calculation can only support binary classification.")
            #if target class not present, throw Exception
            else:
                raise Exception("Error - target class is not present")
            return auc_target