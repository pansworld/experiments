# -*- coding: utf-8 -*-

"""
Created on Wed Sep  14 20:51:05 2023
Loss class that defines the various losses to use during the optimization process.
@author: PSharma
"""

import numpy as np
from enum import Enum

class LossFuction(Enum):
    MEAN_ABSOLUTE_ERROR=0
    MSE=1
    BINARY_CROSSENTROPY=2
    MULTICLASS_CROSSENTROPY=3
    


class Loss():
    def __init__(self):
        self.total_loss=0
        self.loss_history=[]
        self.debug=False
        self.epsilon=1e-5
        
    def calcLoss(self, y_pred, y_actual):
        NotImplemented

class LogCoshLoss(Loss):
    def __init__(self):
        super().__init__()
        self.name = 'LogCoshLoss'
        
    def calcLoss(self, y_pred, y_actual):
        err = y_actual - y_pred
        loss = np.log((np.exp(err) + np.exp(-err))/2)
        self.total_loss = np.mean(np.sum(loss))
        self.loss_history.append(self.total_loss)
        return self.total_loss
    
    def derivative(self, y_pred, y_actual):
        err = y_actual - y_pred
        gradient = (np.exp(2*err)-1)/(np.exp(2*err)+1)
        return gradient


#Relative Square Error Loss
class MSELogLoss(Loss):
    def __init__(self):
        super().__init__()
        self.name = 'MSELogLoss'
        
    
    def getepsilonAdjustedValue(self, data):
        data = np.hstack((data,np.ones(data.shape)*self.epsilon)).reshape(2,data.shape[1])
        data = np.amax(data.T, axis=1, keepdims=True).T
        return data
        
    def calcLoss(self, y_pred, y_actual):
        y_actual = self.getepsilonAdjustedValue(y_actual)
        y_pred = self.getepsilonAdjustedValue(y_pred)

        #y_pred = np.max(np.array(y_pred), np.ones(y_pred.shape[0])*self.epsilon, axis=0)
        #y_actual = np.max(np.array(y_actual), self.epsilon)
        self.total_loss = np.mean(np.square((np.log(y_actual + 1.0) - np.log(y_pred + 1.0))))
        self.loss_history.append(self.total_loss)
        return self.total_loss
    
    def derivative(self, y_pred, y_actual):
        y_actual = self.getepsilonAdjustedValue(y_actual)
        y_pred = self.getepsilonAdjustedValue(y_pred)

        gradient = -2*(np.log(y_actual+1) - np.log(y_pred + 1))/(y_pred + 1)
        return gradient



#Relative Square Error Loss
#Does not work with Regression
class RSELoss(Loss):
    def __init__(self):
        super().__init__()
        self.name = 'RSELoss'
        
    def calcLoss(self, y_pred, y_actual):
        y_mean = y_actual.mean()
        self.total_loss = np.sum(np.square(y_actual - y_pred))/(np.sum(np.square(y_actual - y_mean))+1e-1)
        self.loss_history.append(self.total_loss)
        #print(f'Num: {np.sum(np.square(y_actual - y_pred))}  Den: {(np.sum(np.square(y_actual - y_mean))+1e-5)}  Self Loss: {self.total_loss}')
        return self.total_loss
    
    def derivative(self, y_pred, y_actual):
        y_mean = y_actual.mean()
        gradient = -2*((y_mean-y_actual)*(y_pred-y_actual))/np.power((y_actual-y_mean),3)
        return gradient


#Relative Square Error Loss
#Does not work with Regression
class RRMSELoss(Loss):
    def __init__(self):
        super().__init__()
        self.name = 'RRMSELoss'
        
    def calcLoss(self, y_pred, y_actual):
        #print(f'y_pred: {y_pred} y_actual: {y_actual} Num: {np.sum(np.square(y_actual - y_pred))} Den: {np.sum(np.square(y_pred+1e-5))}')
        self.total_loss = np.sqrt(np.sum(np.square(y_actual - y_pred))/np.sum(np.square(y_pred)))
        self.loss_history.append(self.total_loss)
        #print(f'Num: {np.sum(np.square(y_actual - y_pred))}  Den: {(np.sum(np.square(y_actual - y_mean))+1e-5)}  Self Loss: {self.total_loss}')
        return self.total_loss
    
    def derivative(self, y_pred, y_actual):
        gradient = (y_actual*(y_pred-y_actual))/(y_pred*np.abs(y_pred)*np.abs(y_pred-y_actual))
        return gradient


#Modified Huberloss
class ModifiedHuberLoss(Loss):
    def __init__(self):
        super().__init__()
        self.name = 'ModifiedHuberLoss'
        self.delta = 0.95
        
    def calcLoss(self, y_pred, y_actual):
        #self.delta = 1/(1 + np.exp(-np.abs(np.std(y_actual)-np.std(y_pred))))
        self.total_loss = self.delta*np.sum(np.square(y_actual-y_pred)) + (1-self.delta)*(np.sum(np.abs(y_actual - y_pred)))
        self.loss_history.append(self.total_loss)
        #print(f'Num: {np.sum(np.square(y_actual - y_pred))}  Den: {(np.sum(np.square(y_actual - y_mean))+1e-5)}  Self Loss: {self.total_loss}')
        return self.total_loss
    
    def derivative(self, y_pred, y_actual):
        #self.delta = 1/(1 + np.exp(-np.abs(np.std(y_actual)-np.std(y_pred))))
        gradient = self.delta*(y_actual - y_pred) + (1 - self.delta)*(y_actual - y_pred)/np.abs((y_actual - y_pred))
        return gradient


#Huberloss
class HuberLoss(Loss):
    def __init__(self):
        super().__init__()
        self.name = 'ModifiedHuberLoss'
        self.delta = 2.75
        
    def calcLoss(self, y_pred, y_actual):
        huber_mse = 0.5 * np.square(y_actual - y_pred)
        huber_mae = self.delta * (np.abs(y_actual - y_pred) - 0.5 * (np.square(self.delta)))
        #self.delta = 1/(1 + np.exp(-np.abs(np.std(y_actual)-np.std(y_pred))))
        self.total_loss = np.where(np.abs(y_actual - y_pred) <= self.delta, huber_mse, huber_mae)
        self.loss_history.append(self.total_loss)
        #print(f'Num: {np.sum(np.square(y_actual - y_pred))}  Den: {(np.sum(np.square(y_actual - y_mean))+1e-5)}  Self Loss: {self.total_loss}')
        return self.total_loss
    
    def derivative(self, y_pred, y_actual):
        huber_mse_gradient = (y_actual - y_pred)
        huber_mae_gradient = self.delta*(y_actual - y_pred)/(np.abs((y_actual - y_pred)))
        #self.delta = 1/(1 + np.exp(-np.abs(np.std(y_actual)-np.std(y_pred))))
        gradient = np.where(np.abs(y_actual - y_pred) <= self.delta, huber_mse_gradient, huber_mae_gradient)
        return gradient



#Mean Square Error Loss
class MSELoss(Loss):
    def __init__(self):
        super().__init__()
        self.name = 'MSELoss'
        
    def calcLoss(self, y_pred, y_actual):
        self.total_loss = (1/(2*y_pred.shape[1]))*np.sum(np.square(y_actual - y_pred))
        self.loss_history.append(self.total_loss)
        return self.total_loss
    
    def derivative(self, y_pred, y_actual):
        #print(f'Loss: {(y_actual - y_pred)}')
        gradient = (y_actual - y_pred)
        return gradient
        
#Binary Cross Entropy Loss
class BinaryCrossEntropy(Loss):
    def __init__(self):
        super().__init__()
        self.name = 'BinaryCrossEntropy'
        
    def calcLoss(self, y_actual, y_pred):
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if self.debug:
            print('In BinaryCrossEntropy.calcLoss')
            print(f'Actual = {y_actual}')
            print(f'Predicted = {y_pred}')
            print(np.log(y_pred))
            print(np.log(1-y_pred))
        
        self.total_loss = - (1/(y_pred.shape[1]))*np.sum(y_actual*np.log(y_pred) + (1-y_actual)*np.log(1-y_pred))
        self.loss_history.append(self.total_loss)
        return self.total_loss
    
    def derivative(self, y_pred, y_actual):
        y_pred = np.clip(y_pred, 1e-7, 1-(1e-7))
        
        if self.debug:
            print('In BinaryCrossEntropy.derivative')
            print(f'y_pred={y_pred} y_actual={y_actual}')
            print(f'((1-y_actual)/(1-y_pred))= {((1-y_actual)/(1-y_pred))}')
            print(f'(y_actual/y_pred)= {(y_actual/y_pred)}')

        gradient = (y_actual - y_pred)/((y_pred)*(1-y_pred))
        return gradient
    
    
