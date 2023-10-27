# -*- coding: utf-8 -*-

"""
Created on Wed Sep  14 20:51:05 2023
Optimizer class that defines the various optimizations to use.
@author: PSharma
"""
from enum import Enum
import numpy as np
from ActivationFunctions import ActivationType


class OptimizerType(Enum):
    GRADIENT_DESCENT=0
    STOCHASTIC_GRADIENT_DESCENT=1
    MINIBATCH_GRADIENT_DESCENT=2
    DISTRIBUTIONALLY_ROBUST=4
    
class GradientDescentCalc(Enum):
    STOCHASTIC=0
    ADAM=1


class Optimizer():
    def __init__(self):
        self.optimizer_type=None
        self.loss=0
        self.debug=False
        self.batch_size=0
        self.train_loss=0
        self.epoch_accuracy = 0
        self.epoch_train_accuracy = 0
        self.iteration = 0
    
    def getregularization(self, model, layer):
        loss_penalty_term=0
        grad_penalty_term=0
        
        if model.regularization == 'L2':
            loss_penalty_term = np.sum(np.square(layer.weights))
            grad_penalty_term = np.sum(layer.weights)
            
        return loss_penalty_term, grad_penalty_term
        

    def maxpooling(self, gradient, layer_input_a):
        print(gradient.shape)
        print(layer_input_a.shape)
        argm = np.argmax(layer_input_a, axis=0).reshape(1, gradient.shape[1])
        print(argm.shape)
        max_grad = gradient[:, argm[0]]
        print(max_grad)
        return np.where(gradient == max_grad, max_grad, 0)



    def rungradientstep(self, x_batch, y_batch, model, last_layer_index, iteration=0, y_mean=0):
        
        
        if self.debug:
            print(f'x_batch {x_batch.shape}')
            print(f'i = {x_batch.shape} {y_batch.shape}')

        #Run the forward pass
        model.forwardPass(x_batch)
        
        
        grad_penalty_term =0
        train_loss = 0
        
        curr_layer_index = last_layer_index

        #Revese back from the laters
        for layer in reversed(model.model):
            if self.debug:
                layer.debug = True
                
            loss_penalty_term, grad_penalty_term = self.getregularization(model, layer)
            
            #Is this y_pred
            if curr_layer_index == last_layer_index:
                ###Get the data from the prior layer
                #Get prior layer
                prior_hidden_layer = model.model[curr_layer_index-1]
                prior_output_a= prior_hidden_layer.output_a
                
                #Calculate the loss
                if self.debug:
                    model.loss.debug=True
                    print(f'Before Loss Calc: y_batch {y_batch}')
                    print(f'Before Loss Calc: layer.output_a {layer.output_a}')
                
                #print(f'Loss Components: {model.loss.calcLoss(y_batch, layer.output_a).mean()} {model.reg_lambda} {model.reg_lambda*loss_penalty_term.mean()}')
                #print(f'y_batch: {y_batch}')
                train_loss = model.loss.calcLoss(layer.output_a, y_batch).mean() - model.reg_lambda*loss_penalty_term.mean()
                if self.debug:
                    print(self.train_loss)
                
                if model.loss.name == 'BinaryCrossEntropy':
                    y_pred = (layer.output_a > 0.5).astype(int)
                    self.epoch_train_accuracy += (y_batch == y_pred).sum()/y_batch.shape[1]
                elif model.loss.name == 'MSELoss':
                    y_pred = layer.output_a

                #Adjust the weights
                if layer.activation == ActivationType.SIGMOID:
                    delta = model.loss.derivative(layer.output_a, y_batch)*layer.activationObject.sigmoid_derivative(layer.output_a)
                elif layer.activation == ActivationType.RELU:
                    delta = model.loss.derivative(layer.output_a, y_batch)*layer.activationObject.relu_derivative(layer.output_a)
                
                #Run the delta through an optimizer for get the final delta
                layer.delta = delta
                model.learning_rate, gradient = layer.gradientcalculator.calcgradient(delta, model.learning_rate, time_step=iteration)

                #If Max pooling then adjust the gradient
                if model.maxpooling:
                    gradient = self.maxpooling(gradient, prior_output_a)

                #print(f'After Layer {curr_layer_index} {model.reg_lambda}')

                #layer.weights = layer.weights + model.learning_rate*np.dot(delta,prior_output_a.T)*(1/y_batch.shape[1]) - model.reg_lambda*grad_penalty_term*(1/y_batch.shape[1])
                #print(f'Layer {curr_layer_index} delta: {delta.sum()} Gradient: {gradient.sum()} Dot Product:{np.dot(gradient,prior_output_a.T).shape} Layer Weights Shape: {layer.weights.shape} Prior Output Shape: {prior_output_a.shape[1]}')
                layer.weights = layer.weights + np.dot(gradient,prior_output_a.T)*(1/y_batch.shape[1]) - model.reg_lambda*grad_penalty_term*(1/y_batch.shape[1])
                layer.bias = layer.bias + model.learning_rate*np.mean(delta)

            #Check of this is the input hidden layer
            elif curr_layer_index == 0:
                #Get delta of the downstream layer
                next_hidden_layer = model.model[curr_layer_index+1]
                next_delta = next_hidden_layer.delta
                next_weights = next_hidden_layer.weights
                #print(f'curr_layer_right_index Next Layer Delta {next_delta.shape}')
                
                #Adjust the weights
                if layer.activation == ActivationType.SIGMOID:
                    delta = np.dot(next_weights.T, next_delta)*layer.activationObject.sigmoid_derivative(layer.output_a)
                elif layer.activation == ActivationType.RELU:
                    delta = np.dot(next_weights.T, next_delta)*layer.activationObject.relu_derivative(layer.output_a)
                
                layer.delta = delta
                model.learning_rate, gradient = layer.gradientcalculator.calcgradient(delta, model.learning_rate, time_step=iteration)

                #If Max pooling then adjust the gradient
                if model.maxpooling:
                    gradient = self.maxpooling(gradient, x_batch)


                #print(f'Layer {curr_layer_index} delta: {delta.sum()} Gradient: {gradient.sum()} Dot Product:{np.dot(gradient,x_batch.T).shape} Layer Weights Shape: {layer.weights.shape} Prior Output Shape: {prior_output_a.shape[1]}')
                #layer.weights = layer.weights + model.learning_rate*np.dot(delta,x_batch.T)*(1/prior_output_a.shape[1]) - model.reg_lambda*grad_penalty_term*(1/y_batch.shape[1])
                layer.weights = layer.weights + np.dot(gradient,x_batch.T)*(1/prior_output_a.shape[1]) - model.reg_lambda*grad_penalty_term*(1/y_batch.shape[1])
                layer.bias = layer.bias + model.learning_rate*np.mean(delta)
            else:
                #Get delta and weights of the downstream layer
                next_hidden_layer = model.model[curr_layer_index+1]
                next_delta = next_hidden_layer.delta
                next_weights = next_hidden_layer.weights
                
                #Get the output of the upstream layer
                prev_hidden_layer = model.model[curr_layer_index-1]
                prev_layer_output = prev_hidden_layer.output_a
                
                #Adjust the weights
                if layer.activation == ActivationType.SIGMOID:
                    delta = np.dot(next_weights.T, next_delta)*layer.activationObject.sigmoid_derivative(layer.output_a)
                elif layer.activation == ActivationType.RELU:
                    delta = np.dot(next_weights.T, next_delta)*layer.activationObject.relu_derivative(layer.output_a)
                
                layer.delta = delta
                model.learning_rate, gradient = layer.gradientcalculator.calcgradient(delta, model.learning_rate, time_step=iteration)

                #If Max pooling then adjust the gradient
                if model.maxpooling:
                    gradient = self.maxpooling(gradient, prev_layer_output)


                #print(f'Layer {curr_layer_index} delta: {delta.sum()} Gradient: {gradient.sum()} Dot Product:{np.dot(gradient,prev_layer_output.T).shape}  Layer Weights Shape: {layer.weights.shape} Prior Output Shape: {prior_output_a.shape[1]}')
                #layer.weights = layer.weights + model.learning_rate*np.dot(delta,prev_layer_output.T)*(1/prior_output_a.shape[1]) - model.reg_lambda*grad_penalty_term*(1/y_batch.shape[1])
                layer.weights = layer.weights + np.dot(gradient,prev_layer_output.T)*(1/prior_output_a.shape[1]) - model.reg_lambda*grad_penalty_term*(1/y_batch.shape[1])
                layer.bias = layer.bias + model.learning_rate*np.mean(delta)

            curr_layer_index -= 1

            
        return train_loss


class AdamOptimizer(Optimizer):
    def __init__(self, beta1=0.9, beta2=0.999, decay_schedule=1):
        super().__init__()
        self.name = "AdamOptimizer"
        self.beta1=beta1
        self.beta2=beta2
        self.m=0
        self.v=0
        self.decay_schedule=decay_schedule
        
    def calcgradient(self, gradient, learning_rate, time_step=1):
        decay = (200 - (200-time_step*self.decay_schedule)) #Mod to the ADAM rule
        decay = max(decay, 1)
        #beta1 = self.beta1**np.exp(self.beta1/decay) #Mod to the ADAM rule
        #beta2 = self.beta2**np.exp(self.beta2/decay)
        beta1 = self.beta1**decay
        beta2 = self.beta2**decay
        eps = 1e-07
        self.m = beta1*self.m + (1-beta1)*gradient
        self.v = beta2*self.v + (1-beta2)*np.square(gradient)
        
        self.m_corr = self.m/(1-beta1)
        self.v_corr = self.v/(1-beta2)
        
        #print(learning_rate*self.m_corr/(np.sqrt(self.v_corr) + eps))
        
        
        return learning_rate, learning_rate*self.m_corr/(np.sqrt(self.v_corr) + eps)
        

class SGDOptimizer(Optimizer):
    def __init__(self):
        super().__init__()
        self.name = "SGDOptimizer"

    def calcgradient(self, gradient, learning_rate, time_step=1):
        #Base Optimizer runs SGD
        return learning_rate, gradient*learning_rate
        
class GradientDescentOptimizer(Optimizer):
    def __init__(self):
        super().__init__()
        self.name = "GradientDescentOptimizer"

    def optimize(self, x, y, model, iteration=0, batch_size=0):
        last_layer_index = len(model.model)-1
        self.train_loss=0
        
        
    
        i = 0
        total_data_size = x.shape[1]

        while i < (total_data_size // batch_size) and (i+1)*batch_size < y.shape[1]:
            batch_start = (i)*batch_size
            batch_end = (i+1)*batch_size
            x_batch = x[:, batch_start:batch_end ]
            y_batch = y[:, batch_start:batch_end ]
            
            #Run the gradient pass
            self.train_loss += self.rungradientstep(x_batch, y_batch, model, last_layer_index, iteration=iteration)
            #print(f'y_batch: {y_batch}')


            i += 1

        self.no_of_batches = i
        self.train_loss = self.train_loss/i



class DistributionallyRobustOptimizer(Optimizer):
    def __init__(self):
        super().__init__()
        self.name = "DistributionallyRobustOptimizer"
        self._max_loss_batch_index=[]
        self.early_stop_threshold = 1e-3
        self.ignore_batch=[]
        self.no_of_batches=0

    def optimize(self, x, y, model, iteration=0, batch_size=8, method=1):
        
        _train_loss = []
        _batch_data = [] #Has the data for a particular batch
        last_layer_index = len(model.model)-1


        i = 0

        if method == 0:
            #First break it up into different distributions based on data
            y_data_histogram = np.histogram(y, range=[0, np.max(y)])
            no_of_distributions = len(y_data_histogram[0])
            
            if self.debug:
                print(y_data_histogram)
                print(f'Y Bins:  {y_data_histogram[1]} Y Distribution: {y_data_histogram[0]/np.sum(y_data_histogram[0])}')
            
    
            #each distribution represents a batch. It may or may not match a batch size
            #By default we do not adhere to batches
            #Run through each set and see which one has the maximum loss
            #Use the one with the maximum loss to optimize and run a gradient descent to calculate weights and bias
            y_range = y_data_histogram[1]
            self.no_of_batches = 0


            while i < no_of_distributions:
                #Form the batches
                batch_indices=[]
                indices = np.where((y > y_range[i]) & (y < y_range[i+1]))[1]
                if (len(indices) > 0):
                    y_dist = y[:,indices]
                    x_dist = x[:,indices]
                    
                    #Randomly select samples from the data for a batch size
                    if batch_size > 0:
                        batch_indices = np.random.choice(y_dist.shape[1], batch_size)
                        #print(f'Batch Indices {batch_indices}')
                        x_batch = x_dist[:, batch_indices]
                        y_batch = y_dist[:, batch_indices]
                    else:
                        x_batch = x_dist
                        y_batch = y_dist
                    
                    _batch_data.append((y_batch, x_batch))
                    
                    model.forwardPass(x_batch)
                    
                    #Regularization
                    output_layer = model.model[-1]
                    #loss_penalty_term, grad_penalty_term = self.getregularization(model, output_layer)
                    #_train_loss.append(model.loss.calcLoss(y_batch, output_layer.output_a).mean() - model.reg_lambda*loss_penalty_term.mean())
                    _train_loss.append(model.loss.calcLoss(y_batch, output_layer.output_a).mean())

                    self.no_of_batches += 1

                i += 1
            
            
            
        else:
            #Using the batch_Size create multiple batches
            total_data_size = y.shape[1]
            
            
            ##We break it into batches and only process the batch with the highest loss
            ###
            #TODO: Optimize to use batching
            #self.no_of_batches = total_data_size // batch_size
            #x_b = np.reshape(x[:, :self.no_of_batches*batch_size], (self.no_of_batches,x.shape[0],batch_size))
            #y_b = np.reshape(x[:, :self.no_of_batches*batch_size], (self.no_of_batches,x.shape[0],batch_size))
            #print(x_b.shape, y_b.shape)
            while i < (total_data_size // batch_size):
                batch_start = (i)*batch_size
                batch_end = (i+1)*batch_size
                x_batch = x[:, batch_start:batch_end ]
                y_batch = y[:, batch_start:batch_end ]
                _batch_data.append((y_batch, x_batch))
                
                
                model.forwardPass(x_batch)
                #Regularization
                output_layer = model.model[-1]
                #loss_penalty_term, grad_penalty_term = self.getregularization(model, output_layer)
                #_train_loss.append(model.loss.calcLoss(y_batch, output_layer.output_a).mean() - model.reg_lambda*loss_penalty_term.mean())
                _train_loss.append(model.loss.calcLoss(y_batch, output_layer.output_a).mean())
                
                i += 1

            self.no_of_batches = i
        


        ####Keep going until you have run through all the batches.
        '''
        while (len(np.unique(self._max_loss_batch_index)) != self.no_of_batches):
            i=0
            while(i < self.no_of_batches):
                y_batch, x_batch = _batch_data[i]
                model.forwardPass(x_batch)
                #Regularization
                output_layer = model.model[-1]
                #loss_penalty_term, grad_penalty_term = self.getregularization(model, output_layer)
                #_train_loss.append(model.loss.calcLoss(y_batch, output_layer.output_a).mean() - model.reg_lambda*loss_penalty_term.mean())
                if (i not in self.ignore_batch):
                    _train_loss.append(model.loss.calcLoss(y_batch, output_layer.output_a).mean())
    
                i += 1
        '''

        #Re-order batch and run through the batch in descending order of loss
        i=0
        while(i < self.no_of_batches):
            #Batch to run gradient.
            grad_index = np.argmax(_train_loss)
            self._max_loss_batch_index.append(grad_index)
            y_batch, x_batch = _batch_data[grad_index]
            self.train_loss = _train_loss[grad_index]
            #print(f'Batch Index: {len(np.unique(self._max_loss_batch_index))} loss={self.train_loss}')
        
            #Run a single gradient pass on the max loss batch
            self.rungradientstep(x_batch, y_batch, model, last_layer_index, iteration=iteration)
            
            #if (len(_train_loss)>5):
            #    train_diff = _train_loss[-1] - _train_loss[-5].mean()
            #    print(f'{train_diff}')
            #    if (abs(train_diff) < abs(self.early_stop_threshold) or train_diff > 0 ):
            #       self.ignore_batch.append(grad_index)
            i += 1
            
            #Go to the next batch
            self.train_loss = _train_loss[grad_index]
            del _train_loss[grad_index]

            
        _train_loss=[]
        #print(f'--------------{self.train_loss}------------------')


        self.no_of_batches=0
        self._max_loss_batch_index=[]
        
'''
        #Batch to run gradient.
        grad_index = np.argmax(_train_loss)
        self._max_loss_batch_index.append(grad_index)
        y_batch, x_batch = _batch_data[grad_index]
        self.train_loss = _train_loss[grad_index]
        print(np.unique(self._max_loss_batch_index))
        print(self.no_of_batches)

        #Run a single gradient pass on the max loss batch
        self.rungradientstep(x_batch, y_batch, model, last_layer_index, iteration=iteration)
'''        
    