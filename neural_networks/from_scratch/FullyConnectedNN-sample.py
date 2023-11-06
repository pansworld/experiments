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

from ActivationFunctions import ActivationType
from Normalizers import MinMax
import matplotlib.pyplot as plt
import numpy as np
from Model import Model
from ProjectsDataProcessor import ProjectDataProcessor
from scipy.stats import skew

def reject_outliers(data, m=2):
    without_outliers = data[:, -1] - data[:, -1].mean() < m*np.std(data[:, -1])
    retval = data[without_outliers, :]
    print(f'Data Shape {data.shape} Without Outliers {retval.shape}')
    return retval


class FullyConnectedNN(Model):
    def __init__(self, model_spec, debug=False, save_model=False, version=1, overwrite=False):
        super().__init__(model_spec, debug, save_model, version, overwrite)
        self.model_spec = model_spec
        self.name = "FullyConnectedNN"
        
    def getRegressionAccuracy(self, y_actual, data, data_scaler=None, print_stats=True, title=""):
        y_predicted = self.predict(data, 1)

        if data_scaler is not None:
            y_actual = data_scaler.denormalize(y_actual)
            y_predicted = data_scaler.denormalize(y_predicted)
            
        #Implement accuracy and get the statistics
        accuracy = (np.sum(y_predicted.astype(int)) - np.sum(y_actual))/np.sum(y_actual)
        acc_dist = (y_predicted - y_actual)/y_actual
        acc_dist_mean = acc_dist.mean()
        acc_dist_std = acc_dist.std()
        acc_dist_skewness = skew(acc_dist[0])
        acc_within_range = (acc_dist < accuracy).astype(int)
        
        if print_stats:
            print(f'-----------------------{title} ACCURACT STATS-------------------')
            print(f'{np.sum(acc_within_range)} out of {acc_dist.shape[1]}')
            print(f'Over or Underestimation Percentage: { accuracy }')
            print(f'Percentage dist: { acc_dist }')
            print(f'Accuracy Mean: {acc_dist_mean} Std: {acc_dist_std} Skewness: {acc_dist_skewness}')
            print(f'Actual: {y_actual} Sum: {np.sum(y_actual)} Max: {np.max(y_actual)} Min: {np.min(y_actual)} Mean: {y_actual.mean()} Std: {y_actual.std()}')
            print(f'Predicted {y_predicted} Sum: {np.sum(y_predicted)} Max: {np.max(y_predicted)} Min: {np.min(y_predicted)} Mean: {y_predicted.mean()} Std: {y_predicted.std()}')
            print('-------------------------------------------------------')
        
        return [accuracy, acc_dist, acc_dist_mean, acc_dist_std, acc_dist_skewness, acc_within_range]
        
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
        #last_layer_index = len(self.model)-1
        
        model.optimizer.debug= False
        
        
        for ep in range(0, epochs):
            train_loss = 0
            test_loss=0

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
                
            #Run through optimizer
            model.optimizer.optimize(x_train, y_train, self, iteration=ep+1, batch_size=batch_size)
            epoch_train_accuracy += model.optimizer.epoch_train_accuracy
            i = model.optimizer.no_of_batches
            train_loss = model.optimizer.train_loss

            self.forwardPass(x_test)

            test_loss = self.loss.calcLoss(y_test, self.model[-1].output_a).mean()
            
            layer = self.model[-1]
            y_pred = (layer.output_a > 0.5).astype(int)
            if self.debug:
                print(y_pred)
                print(y_test)

            if self.loss.name == 'BinaryCrossEntnropy':
                epoch_accuracy = (y_test == y_pred).sum()/y_test.shape[1]
                _accuracy.append(epoch_accuracy)
                _train_accuracy.append(epoch_train_accuracy/(i))

            _train_loss.append(train_loss)
            _test_loss.append(test_loss)
            
            
            #Print the test accuracy
            if self.loss.name == 'BinaryCrossEntropy':
                print(f'Epoch {ep} Train [Loss: {round(train_loss/(i),4)} Accuracy: {round(epoch_train_accuracy,4)}] Test [Loss: {round(test_loss,4)} Accuracy: {round(epoch_accuracy,4)}]')    
            else:
                print(f'Epoch {ep} Train [Loss: {round(train_loss,4)}] Test [Loss: {round(test_loss,4)}]')    
                
            
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
        
        else:
            plt.plot(range(epochs), _test_loss, color="r")
            plt.plot(range(epochs), _train_loss, color="g")
            plt.title(f'Loss (Batch: {batch_size}) LR: {self.learning_rate} REG: {self.regularization} REG_LAMB: {self.reg_lambda}')
            
        plt.show()
        #folder = f'saved_models/{self.name}/{self.version}'
        #plt.savefig(f'{folder}/loss.png', dpi=300)
        
if __name__ == '__main__':
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

    ##############################
    #### Training Cycle
    ##############################
    
    train=False
    model_version=7
    transfer_learning=False
    model = FullyConnectedNN('', debug=False, save_model=True, version=model_version, overwrite=False)

    if train:
        model_spec = '''{"Layers":
                                [
                                    {
                                        "inputsize": 116,
                                        "outputsize": 164,
                                        "activationtype": "RELU",
                                        "initialization": "XAVIER_UNIFORM",
                                        "gradient_calculator": "ADAM"
                                    },
                                    {
                                        "inputsize": 164,
                                        "outputsize": 364,
                                        "activationtype": "RELU",
                                        "initialization": "XAVIER_UNIFORM",
                                        "gradient_calculator": "ADAM"
                                    },
                                    {
                                        "inputsize": 364,
                                        "outputsize": 164,
                                        "activationtype": "RELU",
                                        "initialization": "XAVIER_UNIFORM",
                                        "gradient_calculator": "ADAM"
                                    },
                                    {
                                        "inputsize": 164,
                                        "outputsize": 1,
                                        "activationtype": "RELU",
                                        "initialization": "XAVIER_UNIFORM",
                                        "gradient_calculator": "ADAM"
                                    }
                                ],
                                "losstype": "ModifiedHuber",
                                "learning_rate": 0.001,
                                "regularization": "L2",
                                "regularization_lambda": 0.00001,
                                "optimization": "DRO",
                                "maxpooling": "False"
                           }
                            '''
        
        if transfer_learning:
            model.load_model_version(model_version)
        else:
            model.model_spec = model_spec
            model.buildModel()
    
    
        #Start training the model
        model.train(x,y,epochs=100, batch_size=8, train_test_split=0.8)
        
        #Run the recall segments
        i=0
        for segment in recall_segmented_data:
            x_recall, y_recall = dp.getXandYfromDataFrame(segment, outliers_threshold=outliers_threshold)
            #Run the recall
            model.getRegressionAccuracy(y_recall, x_recall, train_scalers[0][2], title=segment_columns[i])
            i += 1

        
        #Run the validation
        model.getRegressionAccuracy(y_holdout, x_holdout, holdout_rescalers[0][2], title="Holdout Set")
    
        if model.save_model:
            model.save_model_version()

    else:
        model.load_model_version(model_version)
        
        y_predictions = []
        #Run through the prediction segements
        i=0
        for segment in segmented_predict:
            x_predict, y_predict = dp.getXandYfromDataFrame(segment, outliers_threshold=outliers_threshold)
            #Run the recall
            #model.getRegressionAccuracy(y_recall, x_recall, train_scalers[0][2], title=segment_columns[i])
            y_prediction = np.exp(model.predict(x_predict, 1))
            y_predictions.append([segment_columns[i],y_prediction])
            print(f'================={segment_columns[i]} PREDICTION ==================== ')
            print(f'Records: {y_prediction.shape[1]}')
            print(y_prediction.sum())    
            print(y_prediction.mean())    
            print(y_prediction.std())    
            print(y_prediction)    
            print('===================================================================== ')
            i += 1
        
        
        