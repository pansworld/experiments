#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 20:51:05 2023
Singleton class that has the activation functions defined

@author: PSharma
"""
import numpy as np
import unittest
from enum import Enum

class ActivationType(Enum):
    NONE=0
    SIGMOID=1
    TANH=2
    RELU=3
    SOFTMAX=4

class ActivationFunctions:
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(ActivationFunctions, cls).__new__(cls)
        return cls.instance
    
    def sigmoid(self,z, scale_factor=1):
        """Sigmoid Activation Function used for a yes/no problem
        """
        z = z/float(scale_factor)
        return 1/(1+np.exp(-z))
    
    def sigmoid_derivative(self,z, scale_factor=1):
        """Sigmoid Gradient Function used for a yes/no problem
        https://hausetutorials.netlify.app/posts/2019-12-01-neural-networks-deriving-the-sigmoid-derivative/#:~:text=The%20derivative%20of%20the%20sigmoid%20function%20%CF%83(x)%20is%20the,1%E2%88%92%CF%83(x).
        """
        sigmoid_val = self.sigmoid(z, scale_factor)
        return (1/scale_factor)*(sigmoid_val)*(1-sigmoid_val)
    
    def tanh(self,z):
        return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))

    def tanh_derivative(self,z):
        tanh_val = self.tanh(z)
        return 1 - (tanh_val)**2

    
    def relu(self,z):
        zeros = np.zeros(z.shape)
        return np.maximum(zeros, z)
    
    def relu_derivative(self,z):
        derivative = np.where(z > 0, np.ones_like(z), np.zeros_like(z))
        return derivative


    def softmax(self,logits: np.ndarray) -> float:
        logits_exp = np.exp(logits)
        return logits_exp/np.sum(logits_exp, axis=1, keepdims=True)
            

class TestActivationFunctions(unittest.TestCase):
    def setUp(self):
        print("Setting up activations")
        self.activations = ActivationFunctions()
    
    def test_sigmoid(self):
        self.assertEqual(self.activations.sigmoid(0),0.5,"Incorrect Value for 0")
        self.assertAlmostEqual(self.activations.sigmoid(1), 0.7310585786300049,12,"Incorrect value for 1")
        
    def test_tanh(self):
        self.assertEqual(self.activations.tanh(0),0,"Incorrect Value for 0")
        self.assertAlmostEqual(self.activations.tanh(1), 0.7615941559557649,12,"Incorrect value for 1")
        
    def test_relu(self):
        self.assertEqual(self.activations.relu(np.array([0])),0,"Incorrect Value for 0")
        self.assertEqual(self.activations.relu(np.array([-1])), 0,"Incorrect value for -1")
        self.assertEqual(self.activations.relu(np.array([1])), 1,"Incorrect value for 1")     
        
    
    def test_softmax(self):
        self.assertAlmostEqual(np.sum(self.activations.softmax(np.random.rand(1,5))),1,12,"Sum of probabilities is not 1")
        
        
if __name__ == '__main__':
    unittest.main()