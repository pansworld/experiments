# -*- coding: utf-8 -*-

"""
Created on Wed Oct  10 20:51:05 2023
Contains the data processing class for project.
Essentially visualizes the project data and then
Breaks it up into train, test and holdout for validation.
Also preprocesses data for prediction
@author: PSharma

Package Install
pip install imageio
pip install matplotlib
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sb
from Normalizers import MinMax, Mean, LogNormal



class DataProcessor:
    def __init__(self, csv_file, holdout=0):
        self.csv_file = csv_file
        self.holdout = holdout
        self.dataframe = None
        self.y_dataframe = None
        self.x_dataframe = None
        self.readfile()
    
    
    def readfile(self):
        self.dataframe = pd.read_csv(self.csv_file)
        
    
    def getonehotcolumns(self, dataframe, column_name):
        return pd.get_dummies(dataframe['productNumber'])
    
    
    def process(self, y_column_name, column_list=None, onehot_columns_list=None, group_by_list=None):
        df = self.dataframe
        df = df[df['productNumber'].notna()]
        

        #Get the columns that we need
        if column_list is not None:
            df = df[df.columns.intersection(column_list)].fillna(0)
            #Reorder the columns as well
            df = df[column_list]
            
        #Process the onehot columns
        if onehot_columns_list is not None:
            for col_name in onehot_columns_list:
                one_hot = self.getonehotcolumns(df, col_name)
                df = pd.concat([df, one_hot], axis=1, join='inner')
        
        
        if group_by_list is not None:
            df = df.groupby(group_by_list).mean()
            
        df.to_csv('./data/projects/_transformed.csv')
        df = pd.read_csv('./data/projects/_transformed.csv')
        
        self.dataframe = df


    def removecommasfromcolumns(self, df, columns=None):
        if columns is None:
            columns = self.dataframe.columns

        comma_cols = []
        
        
        for col in columns:
            if df[col].dtype == 'object':
                if df[col].str.contains(',').any():
                    comma_cols.append(col)
                
        for col in comma_cols:
            df[col] = df[col].str.replace(',', '')
        
        return df


    def filtercolumns(self, df, filter):
        
        if filter is not None:
            n_cols = df.loc[:, df.columns.str.startswith('CN-')]
            df_first = df.iloc[:, :5]
            df = pd.concat([df_first, n_cols], axis=1, join='inner')
        
        return df


    def selectdata(self, df, criteria=None, criteria_cols=None, drop_criteria_cols=False):

        if criteria is not None:
            df = df[criteria]
            
            if criteria_cols is not None and drop_criteria_cols and df.shape[0] > 0:
                for col in criteria_cols:
                    df = df.drop(col, axis=1)
        
        return df

        
    def getholdoutdata(self, df):
        if self.holdout > 0:
            holdout_indices = np.random.choice(df.shape[0], size=self.holdout)
            holdout_data = df.iloc[holdout_indices]
            
        return holdout_indices, holdout_data
    
    
    def getsegmenteddata(self, df, segment_by_criteria_list=[]):
        segmented_list = []
        if len(segment_by_criteria_list) > 0:
            for criteria in segment_by_criteria_list:
                segmented_list.append(self.selectdata(df, criteria, criteria_cols=None, drop_criteria_cols=False))

        #for i in range(0, len(segmented_list)):
        #    print(f'Segmented List: {segmented_list[i].shape}')
        return segmented_list
    
    def reject_outliers(self, data, m=2):
        """

        Parameters
        ----------
        data : Numpy Array
            Numpy array odata with the y value as the last column
        m : float, optional
            Standard deviation beyond which data is rejected The default is 2.

        Returns
        -------
        retval : Numpy array
            Data without outliers.

        """
        without_outliers = data[:, -1] - data[:, -1].mean() < m*np.std(data[:, -1])
        retval = data[without_outliers, :]
        #print(f'Data Shape {data.shape} Without Outliers {retval.shape}')
        return retval


    def plotdata(self, df, index=-1, bins=None, step=10, outliers_threshold=0):
        
        
        if bins is None:
            bins=np.arange(0,400, step, dtype=int)
    
        curr_plot_data =  df.to_numpy(dtype=float)
        
        #Plot with and without outliers
        plt.hist(curr_plot_data[:, index],bins = bins, color="red")
        
        #Now Reject outliers and plot
        curr_plot_data=self.reject_outliers(curr_plot_data,outliers_threshold )
        plt.hist(abs(curr_plot_data[:, index]),bins = bins, color="green")
        
        plt.legend(["Outliers","Training Data"])
        plt.title(f'{df.columns[index]} Count: {curr_plot_data.shape[0]} Mean {curr_plot_data[:, index].mean()}  Std {curr_plot_data[:, index].std()}')

        plt.show() 

    def getXandYfromDataFrame(self, data_frame, outliers_threshold=0):
        data = data_frame.to_numpy(dtype=float)
        data = self.reject_outliers(data, outliers_threshold)
        x = data[:, 0:-1].T
        y = data[:, -1].reshape(1,data.shape[0])

        return x,y


    
    def plotsgementeddata(self, segment_list, segemented_data, plot_dims=(3,2), bins=None, step=10, outliers_threshold=0):
        
        max_index = len(segment_list)
        curr_index=0
        plot_columns = plot_dims[1]
        plot_rows = plot_dims[0]
        
        
        if bins is None:
            bins=np.arange(0,400, step, dtype=int)

        #Loop for all segements
        while curr_index < max_index:    
            
            #Plot a new figure
            figure, axis = plt.subplots(plot_rows,plot_columns)
            figure.tight_layout(pad=2.0)
            figure.figsize=(8,11)

            for i in range(0,plot_rows):
                for j in range(0, plot_columns):
                    #Increment the index
                    if curr_index == max_index: 
                        break 
                    
                    #print(curr_index, max_index)
                    curr_plot_data =  segemented_data[curr_index].to_numpy(dtype=float)
                    
                    if (curr_plot_data.shape[0] > 0):
                        axis[j,i].hist(abs(curr_plot_data[:, -1]),bins = bins, color="red")
                        
                        #Plot again with the cleaned up data
                        if outliers_threshold > 0:
                            curr_plot_data = self.reject_outliers(curr_plot_data, outliers_threshold)
                            axis[i,j].hist(abs(curr_plot_data[:, -1]),bins = bins, color="green")

                        axis[i,j].set_title(f'{segment_list[curr_index]} \n (mean: {int(curr_plot_data[:, -1].mean())} days) (std dev: {int(curr_plot_data[:, -1].std())}) (Total: {int(curr_plot_data.shape[0])})')
                        axis[i,j].legend(["Outliers","Training Data"])
    
                    curr_index += 1

            plt.show() 
           
    
class ProjectDataProcessor(DataProcessor):
    def __init__(self, csv_file, holdout=0):
        super().__init__(csv_file, holdout)
        self.name = "ProjectDataProcessor"
        
    def process(self, y_column_name, column_list=None, onehot_columns_list=None, group_by_list=None, filter='CN-'):
        super().process(y_column_name, column_list, onehot_columns_list, group_by_list)
        df = self.dataframe
        
        if filter is not None:
            df = self.filtercolumns(df, filter)
            
        df = self.removecommasfromcolumns(df, ['fullTimeEnrollment', 'International_Students_Count'])
        
        #Get the y_column and move it to the end
        y_dataframe = df.pop('duration')
        last_index = len(df.columns)
        df.insert(last_index, 'duration', y_dataframe)
        
        #Drop the project Id columns
        df = df.drop('projectId', axis=1)
        
        self.dataframe = df
    
    def gettraindata(self, segment_by_criteria_list=[], outliers_threshold=0, plot_pearson=True):
        
        df = self.dataframe
        segmented_data = []
        
        criteria_cols = ['projectStage']
        criteria = (df['projectStage'] == 'Delivered') & (df['duration'] > 5)
        df = self.selectdata(df, criteria, criteria_cols, drop_criteria_cols=True)
        
        #Also run the data segmentation logic
        segmented_data = self.getsegmenteddata(df, segment_by_criteria_list)
        
        #Also make sure you have taken out the holdout data
        holdout_indices, self.holdout_data = self.getholdoutdata(df)
        df.drop(df.iloc[holdout_indices].index, inplace=True)
        

        if plot_pearson: self.plotpearson(df,"Raw Data")

        return df, segmented_data
    
    def getdatatopredict(self, segment_by_criteria_list=[]):
        df = self.dataframe
        
        criteria_cols = ['projectStage']
        criteria = (df['projectStage'] == 'Created') | (df['projectStage'] == 'In Process')
        df = self.selectdata(df, criteria, criteria_cols, drop_criteria_cols=True)

        #Also run the data segmentation logic
        segmented_data = self.getsegmenteddata(df, segment_by_criteria_list)
        
        #drop the duration column
        df = df.drop(['duration'], axis=1)
        
        return df, segmented_data
        
    def plotpearson(self, df, title=""):
        C_mat = df.corr(method='pearson')
        C_mat = C_mat.tail(1)
        print(C_mat.transpose().fillna(0).sort_values(by='duration', ascending=False).head(60))
        print(C_mat.transpose().fillna(0).sort_values(by='duration', ascending=True).head(60))
        plt.figure(figsize = (10,1))
        sb.heatmap(C_mat, vmax=0.8, cmap="Blues")
        plt.show()

    def getXandYfromDataFrame(self, data_frame, outliers_threshold=0):
        data = data_frame.to_numpy(dtype=float)

        data = self.reject_outliers(data, outliers_threshold)
        x = data[:, 0:-1].T
        x[:, 2:] = (x[:, 2:]>0).astype(int)
        y = abs(data[:, -1]).reshape(1,data.shape[0])

        return x,y

        
    
    def normalizecolumns(self,df, col_and_type_list):
        """
        Parameters
        ----------
        df : dataframe
            Dataframe to be normalized.
        col_and_type_list : 2 dim array
            Contains the column name and the type of normalization. Supported "mean", "minmax"

        Returns
        -------
        df : Dataframe, rescalers
            Returns the normalized dataframe and scaler objects for columns and the rescaler objects for each column

        """
        col_and_type_list = np.array(col_and_type_list)
        rescalers = []
        
        #Check if we got anything
        if df.shape[0] > 0:
            for row in col_and_type_list:
                if row[1] == "mean":
                    scaler = Mean()
                    
                elif row[1] == "minmax":
                    scaler = MinMax()

                elif row[1] == 'lognormal':
                    scaler = LogNormal()

                data = scaler.normalize(df[row[0]].to_numpy(dtype=float))
                
                df[row[0]] = data

                rescaler = [row[0],row[1] , scaler]
                rescalers.append(rescaler)

        return df, rescalers
            
            
            