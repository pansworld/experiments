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

from Layers import InitializationType, Layer
from ActivationFunctions import ActivationType
from Losses import MSELoss, BinaryCrossEntropy
from Normalizers import MinMax, NormalizerType
import matplotlib.pyplot as plt
import json
import numpy as np
import pdb
from Model import Model

def reject_outliers(data, m=2):
    without_outliers = data[:, -1] - data[:, -1].mean() < m*np.std(data[:, -1])
    retval = data[without_outliers, :]
    print(f'Data Shape {data.shape} Without Outliers {retval.shape}')
    return retval



class LogisticRegression(Model):
    def __init__(self, model_spec, debug=False, save_model=False, version=1):
        super().__init__(model_spec, debug)
        self.model_spec = model_spec
        
        
    def buildModel(self):

    
        self.model_spec_json = json.loads(self.model_spec)    
        #Build the model
        for layerspec in self.model_spec_json['Layers']:
            
            if layerspec['activationtype'] == 'SIGMOID':
                self.model.append(Layer( layerspec['inputsize'],
                                         layerspec['outputsize'],
                                         InitializationType.XAVIER_UNIFORM,
                                         ActivationType.SIGMOID,
                                         debug=False
                                    ))
            elif layerspec['activationtype'] == 'RELU':
                self.model.append(Layer( layerspec['inputsize'],
                                         layerspec['outputsize'],
                                         InitializationType.XAVIER_UNIFORM,
                                         ActivationType.RELU,
                                         debug=False
                                    ))        
        
        #Initialize the loss
        if self.model_spec_json['losstype'] == "MSE":
            self.loss = MSELoss()
        elif self.model_spec_json['losstype'] == "BinaryCrossEntropy":
            self.loss = BinaryCrossEntropy()

        
        #Initialize the data normalizers
        if (self.model_spec_json['datanormalization'] == 'NO_NORMALIZER'):
            self.normalizer=None
            self.has_normalizer=False
        elif (self.model_spec_json['datanormalization'] == 'SCALE_TO_RANGE'):
            self.normalizer=MinMax()
            self.has_normalizer=True
            
        #Initialize the learning rate
        print(self.model_spec_json['learning_rate'])
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
            


    def predict(self, x, y_mean_multiplier):
        """Train
           Expected features shape: (no_of_samples,layer_input_dim)
           Expected label shape: (no_of_samples, layer_output_dim)
        """
        #Run the forward pass
        for layer in self.model:
            layer.forwardPass(x)
            
        print(self.model[-1].output_a)
        print(y_mean_multiplier)
        
        return self.model[-1].output_a*y_mean_multiplier

        

    def train(self, x, y, epochs=1, batch_size=1, train_test_split=0.9):
        """Train
           Expected features shape: (no_of_samples,layer_input_dim)
           Expected label shape: (no_of_samples, layer_output_dim)
           Expected epochs: int
           Expected batch_size: int
           Expected train_test_split: Float
        """
        
        shuffler = np.random.permutation((x.T).shape[0])
        x = x.T[shuffler]
        x = x.T
        y = y.T[shuffler]
        y = y.T
        

        
        
        train_set_size = int(y.shape[1]*train_test_split)
        #test_set_size =  y.shape[1] - train_set_size
        data_size = y.shape[1]


        
        y_train =  y[:, 0:train_set_size]
        x_train = x[:, 0:train_set_size]

        y_test = y[:, train_set_size:data_size]
        x_test = x[:, train_set_size:data_size]

        if self.debug:
            print('Batch Shape')
            print(f'Test X Batch Shape: {x_test.shape}')
            print(f'Test Y Batch Shape: {y_test.shape}')
            print(f'Train X Batch Shape: {x_train.shape}')
            print(f'Train Y Batch Shape: {y_train.shape}')
        
        ep=0
        _train_loss = []
        _accuracy = []
        _train_accuracy = []
        _test_loss = []

        
        for ep in range(0, epochs):
            train_loss = 0
            test_loss=0
            loss_penalty_term =0
            grad_penalty_term =0
            #Break into train and test batch
            #Shuffle data
            shuffler = np.random.permutation((x_train.T).shape[0])
            x_train = x_train.T[shuffler]
            x_train = x_train.T
            y_train = y_train.T[shuffler]
            y_train = y_train.T
            
            total_data_size = x.shape[1]
            epoch_accuracy = 0
            epoch_train_accuracy = 0

            if self.debug:
                print(f'Starting epoch {ep}')
                print(f'{total_data_size // batch_size}')
                
            #Select batch and loop
            i = 0
            while i < (total_data_size // batch_size) and (i+1)*batch_size < y_train.shape[1]:
                batch_start = (i)*batch_size
                batch_end = (i+1)*batch_size
                x_batch = x_train[:, batch_start:batch_end ]
                y_batch = y_train[:, batch_start:batch_end ]
                if self.debug:
                    print(f'x_batch {x_train[:,batch_start:batch_end].shape}')
                    print(f'i = {i} {batch_start} {batch_end} {x_batch.shape} {x_train.shape} {y_batch.shape} {y_train.shape}')
                i = i + 1
                
                #Run the forward pass
                for layer in self.model:
                    layer.forwardPass(x_batch)
                
                #Optimize using backprop
                #Revese back from the laters
                for layer in reversed(self.model):
                    if self.debug:
                        layer.debug = False
                        
                    #Is this y_pred
                    #Calculate the loss
                    if self.debug:
                        self.loss.debug=True
                        print(f'Before Loss Calc: y_batch {y_batch}')
                        print(f'Before Loss Calc: layer.output_a {layer.output_a}')
                    
                    if self.regularization == 'L2':
                        loss_penalty_term = np.sum(np.square(layer.weights))
                        grad_penalty_term = np.sum(layer.weights)

                    train_loss += self.loss.calcLoss(y_batch, layer.output_a).mean() + self.reg_lambda*loss_penalty_term.mean()
                    if self.debug:
                        print(train_loss)
                    
                    if self.loss.name == 'BinaryCrossEntropy':
                        y_pred = (layer.output_a > 0.5).astype(int)
                        epoch_train_accuracy += (y_batch == y_pred).sum()/y_batch.shape[1]
                    elif self.loss.name == 'MSELoss':
                        y_pred = layer.output_a
                    
                    
                    #Adjust the weights
                    if layer.activation == ActivationType.SIGMOID:
                        delta = self.loss.derviative(layer.output_a, y_batch)*layer.activationObject.sigmoid_derivative(layer.output_a)
                    elif layer.activation == ActivationType.RELU:
                        delta = self.loss.derviative(layer.output_a, y_batch)*layer.activationObject.relu_derivative(layer.output_a)

                    layer.delta = delta
                    layer.weights = layer.weights + self.learning_rate*np.dot(delta,x_batch.T)*(1/y_batch.shape[1]) - self.reg_lambda*grad_penalty_term*(1/y_batch.shape[1])
                    layer.bias = layer.bias + self.learning_rate*np.mean(delta)
                            
            #Run the test prediction
            #Calculate Accuracy using the test set
            #Run the forward pass
            for layer in self.model:
                layer.forwardPass(x_test)
            
            test_loss = self.loss.calcLoss(y_test, self.model[-1].output_a).mean() + self.reg_lambda*loss_penalty_term.mean()
            
            layer = self.model[-1]
            y_pred = (layer.output_a > 0.5).astype(int)
            if self.debug:
                print(y_pred)
                print(y_test)

            if self.loss.name == 'BinaryCrossEntropy':
                epoch_accuracy = (y_test == y_pred).sum()/y_test.shape[1]
                _accuracy.append(epoch_accuracy)
                _train_accuracy.append(epoch_train_accuracy/(i))

            _train_loss.append(train_loss/(i))
            _test_loss.append(test_loss)
            
            
            #Print the test accuracy
            if self.loss.name == 'BinaryCrossEntropy':
                print(f'Epoch {ep} Train [Loss: {round(train_loss/(i),4)} Accuracy: {round(epoch_train_accuracy/(i),4)}] Test [Loss: {round(test_loss,4)} Accuracy: {round(epoch_accuracy,4)}]')    
            elif self.loss.name == 'MSELoss':
                print(f'Epoch {ep} Train [Loss: {round(train_loss/(i),4)}] Test [Loss: {round(test_loss,4)}]')    
                
            
            epoch_accuracy = 0
                
        #PLot the outcome
        if self.loss.name == 'BinaryCrossEntropy':
            figure, axis = plt.subplots(2)
            figure.tight_layout(pad=2.0)
            axis[0].plot(range(epochs), _test_loss, color="r")
            axis[0].plot(range(epochs), _train_loss, color="g")
            axis[0].set_title(f'Loss (Batch: {batch_size}) LR: {self.learning_rate} REG: {self.regularization} REG_LAMB: {self.reg_lambda}')
            
            axis[1].plot(range(epochs), _accuracy, color="r")
            axis[1].plot(range(epochs), _train_accuracy, color="g")
            axis[1].set_title('Accuracy') 
        
        elif self.loss.name == 'MSELoss':
            plt.plot(range(epochs), _test_loss, color="r")
            plt.plot(range(epochs), _train_loss, color="g")
            plt.title(f'Loss (Batch: {batch_size}) LR: {self.learning_rate} REG: {self.regularization} REG_LAMB: {self.reg_lambda}')
            
        plt.show()


if __name__ == '__main__':
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
    from imageio import imread
    from PIL import Image
    
    data_size=1000
    images=False
    
    if images:
        dn = MinMax()
        
        
        #Check the file shape
        filename = './data/cats_and_dogs_images/images/Abyssinian_1.jpg'
        img = Image.open(filename).convert('RGB')
        box = (0, 0, 300, 300)
        img2 = img.resize((300,300))
        
        vectorized_picture = np.array(imread(filename))
        vectorized_picture = np.array(img2)
        max = np.max(vectorized_picture)
        min = np.min(vectorized_picture)
    
        vectorized_picture = dn.normalize(vectorized_picture)
        
        ###Data Pre processing
        #Read the list file
        filename = './data/cats_and_dogs_images/annotations/data.txt'
        lines = open(filename).read().splitlines()
        data_sample_raw = np.random.choice(lines, size=(data_size,1))
        x = np.array([])
        y = np.array([])
        
        #Generate Labels and input x
        for line in data_sample_raw:
    
            name = line[0].split()[0]
    
            filename = f'./data/cats_and_dogs_images/images/{name}.jpg'
            img = Image.open(filename).convert('RGB')
            img = img.resize((300,300))
            vectorized_picture = np.array(imread(filename))
            vectorized_picture = np.array(img).ravel()
            vectorized_picture = vectorized_picture.reshape((vectorized_picture.shape[0],1))
            vectorized_picture = dn.normalize(vectorized_picture)
            
            x = np.append(x, vectorized_picture)
            
            if name.istitle():
                y = np.append(y, [1])
            else:
                y = np.append(y,[0])
        
        
        #Reshape y
        y = y.reshape((1,y.shape[0]))
        x = x.reshape((vectorized_picture.shape[0],data_size))
    
    else:
        #Generate dummy data
        x = np.random.rand(40,data_size)
        #y = np.random.rand(1,35)
        y =  np.random.randint(0,2,size=(1,data_size))
        
        #Projects
        
        df = pd.read_csv('./data/projects/project-delivery-2022-2023.csv')
        df = df.drop('Project Id', axis=1)
        data = df.to_numpy(dtype=float)
        
        
        
        plt.hist(abs(data[:, -1]),bins = [0, 50, 100, 150, 200, 250, 300, 350, 400])
        print(data[:, -1].mean())
        
        #Reject outliers        
        data = reject_outliers(data, 3)
        #outliers = data[:, -1] - data[:, -1].mean() < 3*np.std(data[:, -1])
        #data = data[outliers, :]
        plt.hist(abs(data[:, -1]),bins = [0, 50, 100, 150, 200, 250, 300, 350, 400])
        plt.title(f'All Projects mean: {int(data[:, -1].mean())}')
        plt.show()
  
        
        ##Get all the project for CN-PSOFTINTEG
        dt = df[df['CN-ERPINTEGRATE']>0].to_numpy(dtype=float)
        cn_integrate = reject_outliers(dt, 2)
        figure, axis = plt.subplots(3,2)
        figure.tight_layout(pad=2.0)
        figure.figsize=(8,11)
        axis[0,0].hist(abs(cn_integrate[:, -1]),bins = [0, 50, 100, 150, 200, 250, 300, 350, 400])
        axis[0,0].set_title(f'CN-ERPINTEGRATE (mean: {int(cn_integrate[:, -1].mean())} days) (Total: {cn_integrate.shape[0]})')
        print(f'CN-ERPINTEGRATE \t {abs(cn_integrate[:, -1].mean())}')

        dt = df[df['CN-BANNERINTEG']>0].to_numpy(dtype=float)
        cn_integrate = reject_outliers(dt, 2)
        axis[0,1].hist(abs(cn_integrate[:, -1]),bins = [0, 50, 100, 150, 200, 250, 300, 350, 400])
        axis[0,1].set_title(f'CN-BANNERINTEG (mean: {int(cn_integrate[:, -1].mean())} days) (Total: {cn_integrate.shape[0]})')
        print(f'CN-BANNERINTEG \t {cn_integrate[:, -1].mean()} {cn_integrate.shape[0]}')
        
        dt = df[df['CN-WDINTEG']>0].to_numpy(dtype=float)
        cn_integrate = reject_outliers(dt, 2)
        axis[1,0].hist(abs(cn_integrate[:, -1]),bins = [0, 50, 100, 150, 200, 250, 300, 350, 400])
        axis[1,0].set_title(f'CN-WDINTEG (mean: {int(cn_integrate[:, -1].mean())} days) (Total: {cn_integrate.shape[0]})')
        print(f'CN-WDINTEG \t {cn_integrate[:, -1].mean()}')

        dt = df[df['CN-CMPCDINTEG']>0].to_numpy(dtype=float)
        cn_integrate = reject_outliers(dt, 2)
        axis[1,1].hist(abs(cn_integrate[:, -1]),bins = [0, 50, 100, 150, 200, 250, 300, 350, 400])
        axis[1,1].set_title(f'CN-CMPCDINTEG (mean: {int(cn_integrate[:, -1].mean())} days) (Total: {cn_integrate.shape[0]})')
        print(f'CN-CMPCDINTEG \t {cn_integrate[:, -1].mean()}')
        

        dt = df[df['CN-SPECIALINTEGRAT']>0].to_numpy(dtype=float)
        cn_integrate = reject_outliers(dt, 2)
        axis[2,0].hist(abs(cn_integrate[:, -1]),bins = [0, 50, 100, 150, 200, 250, 300, 350, 400])
        axis[2,0].set_title(f'CN-SPECIALINTEGRAT (mean: {int(cn_integrate[:, -1].mean())} days) (Total: {cn_integrate.shape[0]})')
        print(f'CN-SPECIALINTEGRAT \t {cn_integrate[:, -1].mean()}')

        dt = df[df['CN-PSOFTINTEG']>0].to_numpy(dtype=float)
        cn_integrate = reject_outliers(dt, 2)
        axis[2,1].hist(abs(cn_integrate[:, -1]),bins = [0, 50, 100, 150, 200, 250, 300, 350, 400])
        axis[2,1].set_title(f'CN-PSOFTINTEG (mean: {int(cn_integrate[:, -1].mean())} days) (Total: {cn_integrate.shape[0]})')
        print(f'CN-PSOFTINTEG \t {cn_integrate[:, -1].mean()}')

        plt.savefig('data/projects/project_hist_plot.png', dpi=300)
        plt.show()
        
        
        #Prep the training data
        x = data[:, 0:-1].T
        x[-1, :] = x[-1,:]/x[-1,:].mean()
        x[-2:-1, :] = x[-2:-1,:]/x[-2:-1,:].mean()
        x = x[:,:-10]
        x_holdout = x[:,-10:]
        
        y = abs(data[:, -1]).reshape(1,data.shape[0])
        y = y[:,:-10]
        y_holdout = y[:,-10:]
        y_mean = y.mean()
        y_actual = y[:, -3:-2]
        print(y.mean())
        #y = (y > y.mean()).astype(float)
        y = y/y.mean()
        
        print(x.shape)
        print(y.shape)
    
    logistic_model_spec = '''{"Layers":
                            [
                                {
                                    "inputsize": 100,
                                    "outputsize": 1,
                                    "activationtype": "RELU",
                                    "initialization": "XAVIER_UNIFORM"
                                }
                            ],
                            "losstype": "MSE",
                            "datanormalization": "SCALE_TO_RANGE",
                            "learning_rate": 0.00001,
                            "regularization": "L2",
                            "regularization_lambda": 0.000001
                       }
                        '''
                        
    print(logistic_model_spec)
    
    model = LogisticRegression(logistic_model_spec, debug=False)
    model.buildModel()

    
    model.train(x,y,epochs=15000, batch_size=4, train_test_split=0.8)
    
    print(f'Actual: {y_holdout}')
    y_predicted = model.predict(x_holdout,y_mean)
    print(f'Predicted {y_predicted}')
    print(f'Accuracy: { (y_predicted - y_holdout)/y_actual }')
    
    
    
    
    
    