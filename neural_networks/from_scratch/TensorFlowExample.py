# -*- coding: utf-8 -*-

"""
Created on Wed Oct  10 20:51:05 2023
Contains a sample with tensorflow to compare model from scratch
@author: PSharma

Package Install
conda install tesnorflow

"""

from ActivationFunctions import ActivationType
from Normalizers import MinMax
import matplotlib.pyplot as plt
import numpy as np
import pdb
from Model import Model
import seaborn as sb
from ProjectsDataProcessor import ProjectDataProcessor


from tensorflow import keras
from tensorflow.keras import layers


def createModel(input_shape): 
    X_input = layers.Input(input_shape) 
    X = layers.Dense(144, 'relu')(X_input) 
    X = layers.Dense(384, 'relu')(X) 
    X = layers.Dense(144, 'relu')(X) 
    X_output = layers.Dense(1, 'relu')(X) 
    model = keras.Model(inputs=X_input, outputs=X_output) 
    return model 


if __name__ == '__main__':
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
    from imageio import imread
    from PIL import Image
    
    data_size=1000
    images=False

    ##############################
    #### Data Pre Processing
    ##############################
    #Generate dummy data
    x = np.random.rand(40,data_size)
    #y = np.random.rand(1,35)
    y =  np.random.randint(0,2,size=(1,data_size))


    outliers_threshold=2
    holdout=50

    dp = ProjectDataProcessor('./data/projects/2023-24-raw-project-delivery.csv', holdout=holdout)
    column_list = ['projectId', 'fullTimeEnrollment', 'International_Students_Count','productNumber', 'duration', 'projectStage']
    group_by_list = ['projectId', 'fullTimeEnrollment', 'International_Students_Count', 'projectStage']
    dp.process('duration', column_list, onehot_columns_list=['productNumber'], group_by_list=group_by_list)
    df = dp.dataframe
    
    ###Normalize the columns
    col_mean_normalize_spec = [
                            ["fullTimeEnrollment", "minmax"],
                            ["International_Students_Count", "minmax"]
                        ]
    
    df, global_rescalers = dp.normalizecolumns(df, col_mean_normalize_spec)
    
    
    #Segementation Criteria
    segment_criteria = [     df['CN-ERPINTEGRATE']>0,
                             df['CN-BANNERINTEG']>0,
                             df['CN-WDINTEG']>0,
                             df['CN-CMPCDINTEG']>0,
                             df['CN-SPECIALINTEGRAT']>0,                                                                                   
                             df['CN-PSOFTINTEG']>0,                                 
                             df['CN-COLLEAGUEINT']>0,                                 
                             df['CN-PWRCMPINTEG']>0,                                 
                             df['CN-OLINTERFACE']>0                                
                        ]
    
    segment_columns = ['CN-ERPINTEGRATE', 
                       'CN-BANNERINTEG', 
                       'CN-WDINTEG',
                       'CN-CMPCDINTEG',
                       'CN-SPECIALINTEGRAT',
                       'CN-PSOFTINTEG',
                       'CN-COLLEAGUEINT',
                       'CN-PWRCMPINTEG',
                       'CN-OLINTERFACE'
                       ]
    
    #Get the training data
    train_df, segmented_train = dp.gettraindata(segment_by_criteria_list=segment_criteria, outliers_threshold=outliers_threshold)

    #Plot data
    #dp.plotdata(train_df, outliers_threshold=outliers_threshold)
    #dp.plotsgementeddata(segment_columns, segmented_train, plot_dims=(2,2), outliers_threshold=outliers_threshold)

    ###Rescale the duration
    col_normalize_spec = [["duration", "minmax"]]
    train_df, train_rescalers = dp.normalizecolumns(train_df, col_normalize_spec)

    #Get the holdouts after rescaling
    holdout_data, holdout_rescalers =  dp.normalizecolumns(dp.holdout_data, col_normalize_spec)
    print(holdout_data)
    holdout_data = holdout_data.to_numpy(dtype=float)
    holdout_data = dp.reject_outliers(holdout_data, outliers_threshold)
    y_holdout = holdout_data[:, -1].reshape(1, holdout_data.shape[0])
    x_holdout = holdout_data[:, 0:-1].T
    
    data = train_df.to_numpy(dtype=float)
    data = abs(data)
    
    #print(data.shape)

    data = dp.reject_outliers(data, outliers_threshold)
    x = data[:, 0:-1].T
    x[:, 2:] = (x[:, 2:]>0).astype(int)
    y = abs(data[:, -1]).reshape(1,data.shape[0])
    

    
    #Get the prediction data
    predict_data, segmented_predict = dp.getdatatopredict(segment_by_criteria_list=segment_criteria)
    #Rescale the inputs
    
    
    
    #Extract the prediction data
    predict_df = df[(df['projectStage'] == 'Created') | (df['projectStage'] == 'In Process')]
    predict_df = predict_df.drop('projectStage', axis=1)
    predict_data = predict_df.drop(['duration'], axis=1).to_numpy(dtype=float)
    #train_df = train_df[train_df['duration'] > 5]
    predict_data[:, 2:] =  (predict_data[:, 2:] > 0).astype(int)


    ##############################
    #### Training Cycle
    ##############################
    
    train=True
    model_version=5
    transfer_learning=False
    print(x.shape)
    model = createModel(x.shape[0])


        
