# -*- coding: utf-8 -*-

"""
Created on Wed Sep  7 20:51:05 2023
Model class that defines the Model base class and then
variations of the model
@author: PSharma

Package Install
pip install imageio
pip install matplotlib
"""

import os
import json
from Layers import InitializationType, Layer
from Losses import MSELoss, BinaryCrossEntropy, LogCoshLoss, RSELoss, MSELogLoss, RRMSELoss, ModifiedHuberLoss, HuberLoss
from ActivationFunctions import ActivationType
from Normalizers import MinMax, Normalizer
from Optimizers import DistributionallyRobustOptimizer, GradientDescentOptimizer



class Model:
    def __init__ (self, model_spec, debug=False, save_model=False, version=1, overwrite=False):
        self.layer_zs = None
        self.model = []
        self.model_spec = None
        self.debug=debug
        self.learning_rate=1
        self.version = version
        self.save_model=save_model
        self.overwrite=overwrite
        self.name = "BaseLineModel"
        self.accuracy_calibration = 0.0

    def buildModel(self):
        self.model_spec_json = json.loads(self.model_spec)    
        #Build the model
        for layerspec in self.model_spec_json['Layers']:
            
            
            
            gradient_calculation_method = "STOCHASTIC"

            if layerspec.get('gradient_calculator') is not None:
                if layerspec['gradient_calculator'] == "ADAM":
                    gradient_calculation_method = "ADAM"
                
            
            if layerspec['activationtype'] == 'SIGMOID':
                self.model.append(Layer( layerspec['inputsize'],
                                         layerspec['outputsize'],
                                         InitializationType.XAVIER_UNIFORM,
                                         ActivationType.SIGMOID,
                                         gradient_calc_method=gradient_calculation_method,
                                         debug=False
                                    ))
            elif layerspec['activationtype'] == 'RELU':
                self.model.append(Layer( layerspec['inputsize'],
                                         layerspec['outputsize'],
                                         InitializationType.XAVIER_UNIFORM,
                                         ActivationType.RELU,
                                         gradient_calc_method=gradient_calculation_method,
                                         debug=False
                                    ))        
            if 'SIGMOID_SCALED' in layerspec['activationtype']:
                scale = layerspec['activationtype'].split('_')[-1]
                self.model.append(Layer( layerspec['inputsize'],
                                         layerspec['outputsize'],
                                         InitializationType.XAVIER_UNIFORM,
                                         ActivationType.SIGMOID,
                                         activation_scale=scale,
                                         gradient_calc_method=gradient_calculation_method,
                                         debug=False
                                    ))
        
        #Check if we have maxpooling enabled
        self.maxpooling = False
        if self.model_spec_json['maxpooling'] is not None:
            if self.model_spec_json['maxpooling'] == "True":
                self.maxpooling=True
        
        #Initialize the loss
        if self.model_spec_json['losstype'] == "MSE":
            self.loss = MSELoss()
        elif self.model_spec_json['losstype'] == "BinaryCrossEntropy":
            self.loss = BinaryCrossEntropy()
        elif self.model_spec_json['losstype'] == "LogCosh":
            self.loss = LogCoshLoss()
        elif self.model_spec_json['losstype'] == "RSE":
            self.loss = RSELoss()
        elif self.model_spec_json['losstype'] == "MSELog":
            self.loss = MSELogLoss()
        elif self.model_spec_json['losstype'] == "RRMSE":
            self.loss = RRMSELoss()     
        elif self.model_spec_json['losstype'] == "ModifiedHuber":
            self.loss = ModifiedHuberLoss()     
        elif self.model_spec_json['losstype'] == "Huber":
            self.loss = HuberLoss()     


        #Initialize the data normalizers
        #if (self.model_spec_json['datanormalization'] == 'NO_NORMALIZER'):
        #    self.normalizer=Normalizer()
        #    self.has_normalizer=True
        #elif (self.model_spec_json['datanormalization'] == 'SCALE_TO_RANGE'):
        #    self.normalizer=MinMax()
        #    self.has_normalizer=True
        self.has_normalizer=False
            
        #Initialize the learning rate
        if self.model_spec_json['learning_rate'] is None:
            self.learning_rate = 1
        else:
            self.learning_rate = self.model_spec_json['learning_rate']
            
        
        #Check the type of regularization we are using
        if self.model_spec_json['regularization'] is not None:
            self.regularization=self.model_spec_json['regularization']
            self.reg_lambda=self.model_spec_json['regularization_lambda']
        else:
            self.regularization=None
            self.reg_lambda=0

        if self.model_spec_json['optimization'] == "DRO":
            self.optimizer = DistributionallyRobustOptimizer()
        elif self.model_spec_json['optimization']=="SGD":
            self.optimizer = GradientDescentOptimizer()
        else:
            self.optimizer = GradientDescentOptimizer()

        
        self.printModel()


    
    def save_model_version(self):
        i = 0
        layer_weights_and_bias = []
        for layer in self.model:
            attr_name = f'layer_{i}'
            layer_weights_and_bias.append({attr_name: layer.getWeightsandBiasJson()})
            i=i+1
        
        saved_model = {"model": self.model_spec, "weights_and_bias": layer_weights_and_bias, "accuracy_calibration": self.accuracy_calibration}

        #Check if the version path exists
        folder = f'saved_models/{self.name}/{self.version}'
        if os.path.exists(folder):
            if self.overwrite:
                print("Overwriting previous model.")
            else:
                response = input(f'Overwrite {folder} (y/n)')
                if response == 'y':
                    print("Overwriting previous model.")
                else:
                    print("Did not save model.")
                    return
        else:
            os.makedirs(folder)    
        
        with open(f'{folder}/model.data', "w") as f:
            f.write(json.dumps(saved_model))
            print(f'Saved {folder}/model.data')
        
        
    def load_model_version(self, version):
        folder = f'saved_models/{self.name}/{version}'
        self.version = version
        if os.path.exists(folder):
            print("Loading Model.")
            f = open(f'{folder}/model.data', "r")
            saved_model = json.loads(f.read())
        else:
            print("ERROR: Model data not found.")
            return
        
        #Get the spec and build the model.
        self.model_spec = saved_model["model"]
        self.buildModel()
        
        i = 0
        for layer in self.model:
            attr_name = f'layer_{i}'
            layer.loadWeightsandBiasfromJson(saved_model["weights_and_bias"][i][attr_name])
            i=i+1
        
        #check if saved model has an accuracy calibration
        if saved_model["accuracy_calibration"] is not None:
            self.accuracy_calibration = saved_model["accuracy_calibration"]
        
        print('Loaded Model')
        
        
        
    def train(self, input_data, output_data, epochs=1, batch_size=1):
        NotImplemented

    def optimize(self):
        NotImplemented   

    def printModel(self):
        layer_count = 1
        print('=============================================================================================')
        print(f'Model Spec for {self.name}')
        print('=============================================================================================')
        for layer in self.model:
            print(f'Layer {layer_count} [Weights Shape: {layer.weights.shape}] [Bias Shape: {layer.bias.shape}] [Input Dim: {layer.input_dim}] [Output Dim: {layer.output_dim}]')
            layer_count += 1
            print('----------------------------------------------------------------------------------------------')

    def predict(self, x, y_multiplier=1):
        """Train
           Expected features shape: (no_of_samples,layer_input_dim)
           Expected label shape: (no_of_samples, layer_output_dim)
        """
        #Run the forward pass
        self.forwardPass(x)
 
        retVal = self.model[-1].output_a
            
        
        return retVal*y_multiplier

    def forwardPass(self, x):
        """Train
           Expected features shape: (no_of_samples,layer_input_dim)
        """
        layer_input = x
        for layer in self.model:
            layer.forwardPass(layer_input)
            layer_input = layer.output_a
        