#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 20:51:05 2023
Layer class that defines the base class

@author: PSharma
"""

from enum import Enum
import numpy as np
import unittest
from ActivationFunctions import ActivationType, ActivationFunctions
from Optimizers import Optimizer, SGDOptimizer, AdamOptimizer, GradientDescentCalc
import json
from json import JSONEncoder


class InitializationType(Enum):
    ZEROS=0
    RANDOM=1
    XAVIER_UNIFORM=3


class Initializers:
    
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Initializers, cls).__new__(cls)
        return cls.instance

    def randomInitialization(self, input_dim, output_dim):
        return np.random.rand(input_dim, output_dim)
    
    def xavierUniformInitialization(self, input_dim, output_dim):
        x = np.sqrt(6/(input_dim+output_dim))
        #rng = np.random.default_rng()


        #return rng.uniform(-x,x,size=(input_dim,output_dim))
        return np.random.uniform(-x,x,size=(input_dim,output_dim))


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


class Layer:
    
    def __init__(self, input_dim, output_dim, initialization=InitializationType.RANDOM, activation=ActivationType.SIGMOID, gradient_calc_method="STOCHASTIC", activation_scale=1, debug=False):
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.initializers = Initializers()
        self.debug=debug
        self.activationObject=ActivationFunctions()
        self.activation=activation
        self.output_z=None #Linear output cached
        self.output_a=None #Activation output cached
        self.delta=0 #Gradient for the layer
        self.activation_scale = activation_scale
        
        #Assign the Gradient Calculator
        if gradient_calc_method == "ADAM":
            self.gradientcalculator = AdamOptimizer()
        else:
            self.gradientcalculator = SGDOptimizer()
            
            
        #Initialize weights
        if initialization == InitializationType.XAVIER_UNIFORM:
            self.weights= self.initializers.xavierUniformInitialization(self.output_dim, self.input_dim)
        elif initialization == InitializationType.ZEROS:
            self.weights = np.zeros((self.output_dim, self.input_dim))
        else:
            self.weights = self.initializers.randomInitialization(self.output_dim, self.input_dim)
            
        #Initialize bias
        #self.bias = np.zeros((self.output_dim, 1))
        self.bias = np.zeros((1, 1))
        
        if self.debug:
            print("Parameters and Shapes")
            print(f'Input Dim: {input_dim}')
            print(f'Output Dim: {output_dim}')
            print(f'Weights Shape: {self.weights.shape}')
            print(f'Bias Shape: {self.bias.shape}')
            
    def loadWeightsandBiasfromJson(self, weights_bias_json):
        """
        Parameters
        ----------
        weights_bias_json : STRING
            Json string with the following format.
            {"weights": [...]}, "bias": [...]}

        Returns
        -------
        None.

        """
        input_json = json.loads(weights_bias_json)
        self.weights = np.array(input_json["weights"])
        self.bias = np.array(input_json["bias"])
    
    def getWeightsandBiasJson(self):
        """
        Returns
        -------
        JSON String
            Json string with the following format.
            {"weights": [...]}, "bias": [...]}
        """
        retJson = {"weights": self.weights, "bias": self.bias}
        return json.dumps(retJson, cls=NumpyArrayEncoder)
    
    def forwardPass(self, batch: np.ndarray) -> np.ndarray:
        """Layer forward pass
           Expected batch shape: (layer_input_dim, no_of_samples)
        """
        if self.debug:
            print(f'Batch shape is {batch.shape}')
            print(f'Weights Shape: {self.weights.shape}')
            print(f'Bias Shape: {self.bias.shape}')
            
        self.output_z = np.matmul(self.weights,batch)+self.bias
        if self.debug:
            print(f'Linear output shape is {self.output_z.shape}')
            print(f'Linear output is {self.output_z}')
            

        #Activation logic
        match self.activation:
            case ActivationType.NONE:
                self.output_a = self.output_z
            case ActivationType.SIGMOID:
                self.output_a = self.activationObject.sigmoid(self.output_z, self.activation_scale)
                if self.debug:
                    print('Sigmoid Activation')
            case ActivationType.TANH:
                self.output_a = self.activationObject.tanh(self.output_z)
                if self.debug:
                    print('TANH Activation')
            case ActivationType.RELU:
                self.output_a = self.activationObject.relu(self.output_z)
                if self.debug:
                    print('RELU Activation')
            case ActivationType.SOFTMAX:
                self.output_a = self.activationObject.softmax(self.output_z)
                if self.debug:
                    print('SOFTMAX Activation')
            case default:
                self.output_a = self.output_z
        
        if self.debug:
            print(f'Activation output shape is {self.output_a.shape}')
            print(f'Activation output is {self.output_a}')
        
        
        return self.output_a
    

class TestLayer(unittest.TestCase):
    def setUp(self):
        print("Setting up initializers")
        self.initialiers = Initializers()
        self.input_dim=3
        self.output_dim=2
        self.number_of_samples=32
    
    def test_zero_initialization_layer(self):
        layer_with_zero_init = Layer(self.input_dim,self.output_dim,InitializationType.ZEROS)
        #check the dimennsions
        self.assertTupleEqual(layer_with_zero_init.weights.shape, (self.output_dim, self.input_dim), "Incorrect shape of weights")
        #check the value is zero
        self.assertEqual(np.sum(layer_with_zero_init.weights), 0, "Sum of weights is not zero. Not initialized correctly")

    def test_random_initialization_layer(self):
        layer_with_random_init = Layer(self.input_dim,self.output_dim,InitializationType.RANDOM)
        #check the dimennsions
        self.assertTupleEqual(layer_with_random_init.weights.shape, (self.output_dim, self.input_dim), "Incorrect shape of weights")
        #check the value is zero
        self.assertNotEqual(np.sum(layer_with_random_init.weights), 0, "Sum of weights is zero. Not initialized correctly")

    def test_xavier_uniform_initialization_layer(self):
        layer_with_xavier_init = Layer(self.input_dim,self.output_dim,InitializationType.XAVIER_UNIFORM)
        #check the dimennsions
        self.assertTupleEqual(layer_with_xavier_init.weights.shape, (self.output_dim, self.input_dim), "Incorrect shape of weights")
        #check the value is zero
        self.assertNotEqual(np.sum(layer_with_xavier_init.weights), 0, "Sum of weights is zero. Not initialized correctly")

    def test_layer_forward_pass_activation_none(self):
        layer_with_xavier_init = Layer(self.input_dim,self.output_dim,InitializationType.XAVIER_UNIFORM, activation=None, debug=True)
        x = np.random.rand(self.input_dim, self.number_of_samples)
        #check the dimennsions
        self.assertTupleEqual(layer_with_xavier_init.forwardPass(x).shape, (self.output_dim, self.number_of_samples), "Incorrect shape of weights")
        
    def test_layer_forward_pass_activation_sigmoid(self):
        layer_with_xavier_init = Layer(self.input_dim,self.output_dim,InitializationType.XAVIER_UNIFORM, activation=ActivationType.SIGMOID, debug=True)
        x = np.random.rand(self.input_dim, self.number_of_samples)
        #check the dimennsions
        self.assertTupleEqual(layer_with_xavier_init.forwardPass(x).shape, (self.output_dim, self.number_of_samples), "Incorrect shape of weights")
    
    def test_layer_forward_pass_activation_tanh(self):
        layer_with_xavier_init = Layer(self.input_dim,self.output_dim,InitializationType.XAVIER_UNIFORM, activation=ActivationType.TANH, debug=True)
        x = np.random.rand(self.input_dim, self.number_of_samples)
        #check the dimennsions
        self.assertTupleEqual(layer_with_xavier_init.forwardPass(x).shape, (self.output_dim, self.number_of_samples), "Incorrect shape of weights")

    def test_layer_forward_pass_activation_relu(self):
        layer_with_xavier_init = Layer(self.input_dim,self.output_dim,InitializationType.XAVIER_UNIFORM, activation=ActivationType.RELU, debug=True)
        x = np.random.rand(self.input_dim, self.number_of_samples)
        #check the dimennsions
        self.assertTupleEqual(layer_with_xavier_init.forwardPass(x).shape, (self.output_dim, self.number_of_samples), "Incorrect shape of weights")

    def test_layer_forward_pass_activation_softmax(self):
        layer_with_xavier_init = Layer(self.input_dim,self.output_dim,InitializationType.XAVIER_UNIFORM, activation=ActivationType.SOFTMAX, debug=True)
        x = np.random.rand(self.input_dim, self.number_of_samples)
        #check the dimennsions
        self.assertTupleEqual(layer_with_xavier_init.forwardPass(x).shape, (self.output_dim, self.number_of_samples), "Incorrect shape of weights")

    def test_getWeightsandBiasJson(self):
        layer_with_xavier_init = Layer(self.input_dim,self.output_dim,InitializationType.XAVIER_UNIFORM, activation=ActivationType.SOFTMAX, debug=True)
        #check the dimennsions
        self.assertIn("{\"weights\":", layer_with_xavier_init.getWeightsandBiasJson(), "No weights in json")
        self.assertIn(", \"bias\":", layer_with_xavier_init.getWeightsandBiasJson(), "No bias in json")



        
if __name__ == '__main__':
    unittest.main()
        