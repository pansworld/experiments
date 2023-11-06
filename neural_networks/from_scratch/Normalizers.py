# -*- coding: utf-8 -*-


"""
Created on Wed Sep  20
Model class that defines the Normalizer base class and then
the different type of normalizers to use when loading data
@author: PSharma

"""
import numpy as np
from enum import Enum
import unittest

class NormalizerType(Enum):
    NO_NORMALIZER=0
    SCALE_TO_RANGE=1

class Normalizer:
    def __init__(self):
        self.name = "Base Class Normalizer"
        
    def normalize(self, data):
        return data
        
    def denormalize(self,data):
        return data
                
class MinMax(Normalizer):
    def __init__(self):
        self.name = "MinMax"
        self.max = 0
        self.min = 0
        
    def normalize(self, data):
        """Normalizer using Scale to Range technique
           Expected data shape: (layer_input_dim, no_of_samples) - Not strict
        """
        self.max = np.max(data)
        self.min = np.min(data)
        return (data - self.min)/(self.max-self.min)
    
    def denormalize(self, data):
        return data*(self.max-self.min) + self.min

class LogNormal(Normalizer):
    def __init__(self):
        self.name = "LogNormal"
        
    def normalize(self, data):
        """Normalizer using Scale to Range technique
           Expected data shape: (layer_input_dim, no_of_samples) - Not strict
        """
        return np.log(data + 1e-7)
    
    def denormalize(self, data):
        return np.exp(data)

class Mean(Normalizer):
    def __init__(self):
        self.name = "Mean"
        self.mean = 0.0
        
    def normalize(self, data):
        """Normalizer using Scale to Range technique
           Expected data shape: (layer_input_dim, no_of_samples) - Not strict
        """
        self.mean = data.mean()
        return data/data.mean()
    
    def denormalize(self, data):
        return data*self.mean



class TestNormalizers(unittest.TestCase):
    def setUp(self):
        print("Setting up initializers")
        self.input_dim=3
        self.output_dim=2
        self.number_of_samples=32
        
    def test_scale_to_range_normalizer(self):
        x = np.random.randint(100, size=(self.input_dim, self.number_of_samples))
        scale_to_range_normalizer = MinMax()
        norm_x = scale_to_range_normalizer.normalize(x)
        count_greater_than_1 = (norm_x > 1).sum()
        self.assertLessEqual(count_greater_than_1, 0, "Not normalized correctly by ScaleToRange")
        
        count_less_than_1 = (scale_to_range_normalizer.denormalize(norm_x) > 1).sum()
        self.assertGreaterEqual(count_less_than_1, 0, "Not de-normalized correctly by ScaleToRange")



if __name__ == '__main__':
    unittest.main()