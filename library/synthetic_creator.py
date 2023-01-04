#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 08:46:55 2022

@author: temuuleu
"""

import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
import numpy as np
import time

from datetime import datetime
# test classification dataset
from sklearn.datasets import make_classification
import seaborn as sn

#import pydbgen
import random
import scipy

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score
from timeit import default_timer as timer
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import  GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from collections import Counter

import pickle
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import re
import os

import pandas as pd
import numpy as np
import statsmodels.api as sm
#import plotly.express as px
#import plotly.figure_factory as ff


import numpy as np

# Needed for plotting
import matplotlib.colors
import matplotlib.pyplot as plt


# Needed for generating classification, regression and clustering datasets
import sklearn.datasets as dt

# Needed for generating data from an existing dataset
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

#from ipywidgets import HBox, VBox
from scipy.stats import t
from scipy.stats import norm
from math import inf

import bs4 as bs
import requests


from scipy.stats import gamma
from scipy.stats import norm
from scipy.stats import t
from scipy.stats import beta
import matplotlib.pyplot as plt
import time
from timeit import default_timer as timer
import shap
shap.initjs()
from scipy.stats import chisquare

##dist_list = ['uniform','normal','exponential','lognormal','chisquare','beta']
import sklearn.datasets as dt
from scipy import linalg
import scipy.sparse as sp
# Needed for generating data from an existing dataset
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import numbers
import math
from fractions import Fraction

import math 

from library.learning_lib import create_dir
from library.learning_lib import *


def create_pearson_numerical_data(column, number_of_data, numerical_data_frame, distribution, correlation,
                                  maxiteration=1000, df=1):
    """This code generates a numerical data column with a Pearson correlation coefficient close to a specified value.
    This function generates a random data column using either the normal or chi-squared distribution and
    calculates the Pearson correlation coefficient between the generated data and the column.
    It returns the correlation coefficient, the generated data, and other metadata.


    column:                                 a data column to use as a reference
    number_of_data:                         the number of data points to generate
    numerical_data_frame:                   a data frame containing numerical data
    distribution:                           the distribution to use for generating the data
    correlation:                            the desired Pearson correlation coefficient for the generated data
    maxiteration:                           the maximum number of iterations to perform
    df:                                     the degrees of freedom to use for the chi-squared distribution

    """

    n_length = len(numerical_data_frame.columns)

    def preprocess(i, column, number_of_data):

        tmp_dict = {}
        pearson_correlation = lambda x: abs(correlation - pearsonr(column, x)[0])

        if distribution == "normal":
            data = minimize(pearson_correlation,
                            np.random.randn(number_of_data)).x
        elif distribution == "normal":
            data = minimize(pearson_correlation,
                            np.random.default_rng().chisquare(df, number_of_data)).x

        else:
            data = minimize(pearson_correlation,
                            np.random.rand(number_of_data)).x

        p = pearsonr(column, data)

        temp_numerical_data_frame = pd.concat([numerical_data_frame, pd.DataFrame(data, columns=[n_length])], axis=1)

        covariance_matrix = temp_numerical_data_frame.corr()
        covianves = list(covariance_matrix[n_length][:n_length])
        cov_sum = sum(np.abs(covianves))

        tmp_dict[i] = {}
        tmp_dict[i]["cov_sum"] = cov_sum
        tmp_dict[i]["generated_data"] = data
        tmp_dict[i]["correlation"] = p[0]
        tmp_dict[i]["p"] = p[1]

        return tmp_dict

    # result,tmp_dict =

    tmp_dict = Parallel(n_jobs=-1, verbose=1, pre_dispatch='1.5*n_jobs')(
        delayed(preprocess)(i, column, number_of_data) for i in range(maxiteration))

    result = pd.DataFrame()
    new_tmp_dict = {}

    for i, d in enumerate(tmp_dict):
        index_ = list(d[i].keys())[0]

        p_val = d[i]['p']
        correlation = d[i]['correlation']
        cov_sum = d[i]['cov_sum']
        generated_data = d[i]['generated_data']

        new_tmp_dict[i] = {}

        new_tmp_dict[i]['generated_data'] = generated_data
        new_tmp_dict[i]['p_val'] = p_val
        new_tmp_dict[i]['correlation'] = correlation
        new_tmp_dict[i]['cov_sum'] = cov_sum

        tmp_df = pd.DataFrame(index=(i,))
        tmp_df["i"] = i
        tmp_df["cov_sum"] = cov_sum
        tmp_df["p"] = p_val

        result = pd.concat([result, tmp_df])

    result.sort_values(by=["cov_sum"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    found_index = result.loc[0, "i"]

    target_generated_data = new_tmp_dict[found_index]['generated_data']
    found_p = new_tmp_dict[found_index]['p_val']
    correlation = new_tmp_dict[found_index]['correlation']

    numerical_data_frame = pd.concat([numerical_data_frame, pd.DataFrame(target_generated_data, columns=[n_length])],
                                     axis=1)

    return target_generated_data, correlation, found_p, numerical_data_frame


def create_random_numerical_data(column, number_of_data, numerical_data_frame, distribution, maxiteration=1000, df=1):
    """
    This code defines a function named create_random_numerical_data  for generating a random numerical
    data column with a specified distribution and correlation with a given column in a provided numerical data frame.
    This function generates random data using the specified distribution and calculates the Pearson correlation
     coefficient between the generated data and the column argument. It then concatenates
     the generated data with the numerical_data_frame and calculates the covariance matrix of
     the resulting data frame. The function stores the sum of the absolute values of the covariances between
     the generated data and the existing columns in the numerical_data_frame, as well as the correlation
     coefficient and the p-value of the Pearson test, in a dictionary and returns the dictionary.

    column:                           a column in the numerical data frame
    number_of_data:                   the number of data points to generate
    numerical_data_frame:             a data frame containing numerical data
    distribution:                     the distribution to use for generating the data (either "normal" or "chi-squared")
    maxiteration:                     the maximum number of iterations to perform
    df:                               the degrees of freedom to use for the chi-squared distribution
    """

    n_length = len(numerical_data_frame.columns)

    def preprocess(i, column, number_of_data):

        tmp_dict = {}
        # tmp_df = pd.DataFrame(index = (i,))

        if distribution == "normal":
            data = np.random.randn(number_of_data)
        elif distribution == "normal":
            data = np.random.default_rng().chisquare(df, number_of_data)
        else:
            data = np.random.rand(number_of_data)

        p = pearsonr(column, data)

        temp_numerical_data_frame = pd.concat([numerical_data_frame, pd.DataFrame(data, columns=[n_length])], axis=1)

        covariance_matrix = temp_numerical_data_frame.corr()
        covianves = list(covariance_matrix[n_length][:n_length])
        cov_sum = sum(np.abs(covianves))

        tmp_dict[i] = {}
        tmp_dict[i]["cov_sum"] = cov_sum
        tmp_dict[i]["generated_data"] = data
        tmp_dict[i]["correlation"] = p[0]
        tmp_dict[i]["p"] = p[1]

        return tmp_dict

    # result,tmp_dict =

    tmp_dict = Parallel(n_jobs=-1, verbose=1, pre_dispatch='1.5*n_jobs')(
        delayed(preprocess)(i, column, number_of_data) for i in range(maxiteration))

    result = pd.DataFrame()
    new_tmp_dict = {}

    for i, d in enumerate(tmp_dict):
        index_ = list(d[i].keys())[0]

        p_val = d[i]['p']
        correlation = d[i]['correlation']
        cov_sum = d[i]['cov_sum']
        generated_data = d[i]['generated_data']

        new_tmp_dict[i] = {}

        new_tmp_dict[i]['generated_data'] = generated_data
        new_tmp_dict[i]['p_val'] = p_val
        new_tmp_dict[i]['correlation'] = correlation
        new_tmp_dict[i]['cov_sum'] = cov_sum

        tmp_df = pd.DataFrame(index=(i,))
        tmp_df["i"] = i
        tmp_df["cov_sum"] = cov_sum
        tmp_df["p"] = p_val

        result = pd.concat([result, tmp_df])

    result.sort_values(by=["cov_sum"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    found_index = result.loc[0, "i"]

    target_generated_data = new_tmp_dict[found_index]['generated_data']
    found_p = new_tmp_dict[found_index]['p_val']
    correlation = new_tmp_dict[found_index]['correlation']

    numerical_data_frame = pd.concat([numerical_data_frame, pd.DataFrame(target_generated_data, columns=[n_length])],
                                     axis=1)

    return target_generated_data, correlation, found_p, numerical_data_frame


def create_data_from_column(
        numerical_data_frame,
        data_frame,
        columnname,
        correlation=0.7,
        corr_type='pearsonr',
        data_type="numerical",
        distribution="normal",
        df=1,
        target_p=0.05,
        maxiteration=1000,
        number_of_categories=3,
        categorical_even=True
):
    """This code creates a new data column that is either correlated or not correlated with a given column in a provided data frame.

    numerical_data_frame:           a data frame containing numerical data
    data_frame:                         a data frame containing the data to be used
    columnname:                          the name of the column in data_frame to use as a reference
    correlation :                       the desired Pearson correlation coefficient for the generated data
    corr_type (optional, default "pearsonr"): the type of correlation to use for the generated data ("pearsonr" 
                            for Pearson correlation or "nocorrelation" for no correlation)
    data_type (optional, default "numerical"): the data type of the generated data ("numerical" or "categorical")
    distribution (optional, default "normal"): the distribution to use for generating numerical data
    df (optional, default 1): the degrees of freedom to use for the chi-squared distribution
    target_p (optional, default 0.05): the desired p-value for the generated data
    maxiteration (optional, default 1000): the maximum number of iterations to perform
    number_of_categories (optional, default 3): the number of categories to use for generating categorical data
    categorical_even (optional, default True): whether the categories should be evenly distributed

    """
    # numerical_data_frame
    column = data_frame[columnname]
    # numerical_data_frame              = pd.DataFrame()
    n_length = len(numerical_data_frame.columns)

    without_correlation = 0

    start_time = timer()
    number_of_data = len(column)

    found_min = ''

    # if data_type  == "numerical":

    if corr_type == "nocorrelation":

        data, correlation, found_p, numerical_data_frame = create_random_numerical_data(column,
                                                                                        number_of_data,
                                                                                        numerical_data_frame,
                                                                                        distribution,
                                                                                        maxiteration=maxiteration,
                                                                                        df=df)


    elif corr_type == 'pearsonr':

        data, correlation, found_p, numerical_data_frame = create_pearson_numerical_data(column,
                                                                                         number_of_data,
                                                                                         numerical_data_frame,
                                                                                         distribution,
                                                                                         correlation,
                                                                                         maxiteration=10,
                                                                                         df=df)

    else:
        print("wrong corrtype")


    elapsed_time = round(timer() - start_time, 2)  # in seconds

    return data, elapsed_time, correlation, found_p, numerical_data_frame, found_min


class SingleDataCreater():

    def __init__(self, name="", 
                 number_of_data =100,
                 prob  = 0.5,
                 #data_path = "../data/complette_synthethic",
                 try_label = 1000,
                 label_index = 1):
        
      self.maxiteration               = try_label
      #self.data_path                  = data_path
      self.name                       = name
      self.label_name                 = "label"
      self.columns_created            = False
      self.columns_number             = 1
      self.columns                    = []
      self.datasets                   = {}
      
      self.labels                     = []
      self.number_of_data             = number_of_data
      self.parameter_names            = []
      self.parameter_distribution     = []
      self.parameter_correlations     = []
      self.dataframe                  = pd.DataFrame()
      
      self.ids                        = np.array([i_d for i_d in range(1,number_of_data+1)])
      self.dataframe["id"]            = self.ids
      

      """This code generates an array of binary values as labels using the np.random.choice function from NumPy.
       
       p: the probabilities of selecting the values from [0, 1]. In this case, the probability of 
       selecting a 0 is round(1 - prob, 2) and the probability of selecting a 1 is prob.
       
       For example, if prob is 0.3, the probability of selecting a 0 will be round(1 - 0.3, 2) which is equal to 0.7 
       and the probability of selecting a 1 will be 0.3. This means that the function will generate an array of binary
        values with approximately 70% 0s and 30% 1s.
       """
      labels                              =  np.random.choice([0, 1], 
                                                 size=number_of_data,
                                                  p=[round(1 -prob,2) ,prob])
      
      
      
      self.numerical_data_frame              = pd.DataFrame()
      self.labels                           = labels
      self.dataframe[self.label_name]       = self.labels 
      self.columns                          = []
 
      self.label_index                      = label_index
      self.dataframe["label_index"]         = label_index
      
      #number of dataset
      self.number_of_data                   = len(self.labels)
      #time to create the dataset
      self.all_elapsed_time                 = 0
      db_file_name                          = "data.xlsx"

      #all data
      self.all_data_df                       = pd.DataFrame() 
      self.all_data_df           = pd.concat([self.all_data_df , pd.DataFrame(self.labels , columns=["label"])], axis=1)
      
      #self.dataframe["label_name"]           = self.label_name 

    def plot_label_hist(self):
        """print the distribution of label"""
        self.dataframe[self.label_name].hist() 
        plt.title("label : "+self.label_name)
        plt.show()
        
    def plot_input_hist(self,parameter_name):
        """print the distribution of label"""
        
        self.dataframe[parameter_name].hist() 
        plt.title(parameter_name)
        plt.show()
        
    def show_parameters(self):
        print(f"This function shows the save parameters:")
        print(f"{self.name}")
        print(f"Labelname       :   {self.label_name}")
        print(f"Size            :   {self.number_of_data}")
        print(f"Label Ratio     :   {self.label_ratio}")
        print(f"Parameters      :   {self.parameter_names}")
        print(f"Distribution    :   {self.parameter_distribution}")
        print(f"Correlation     :   {self.parameter_correlations}")
        print(f"Column is created:  {self.columns_created}")
        print(f"Time created in     {self.all_elapsed_time} Seconds ")


    def show_parameters(self):
        
        print("Parameters: ")
        print(f"Parameters:               {self.label_ratio} ")
        print(f"number_of_data:           {self.number_of_data} ")
        print(f"parameter_names:          {self.parameter_names} ")
        print(f"parameter_distribution:   {self.parameter_distribution} ")
        print(f"parameter_correlations:   {self.parameter_correlations} ")
        
        
    def show_covariance(self):
        if self.columns_created:
            self.dataframe[self.parameter_names].hist() 
            plt.show()

            corr = self.dataframe.corr()
            sns.heatmap(corr)
            plt.show()
            
            print("correlation matrix")


    def get_data(self):
        
        return self.all_data_df
    

    def add_collumn(self,
                    new_column_name="",
                    distribution="normal",
                    correlation=0, 
                    corr_type="pearsonr",
                    data_type="",
                    var_type="",
                    from_column_name="",
                    std=10, 
                    expected_value = 100, 
                    r = 2,
                    df = 0,
                    target_p = 0.05,
                    maxiteration = 1000,
                    number_of_categories  = 0,
                    categorical_even   = True
                    ):    
        
        """add new columns to the dataset"""
        if correlation == 0:
            corr_type="nocorrelation"

            
        if  not from_column_name in  self.dataframe.columns:
            print("the source columns doesn't exist")
            return 0
        
        
        col_name              = str(self.columns_number)+ "_"+new_column_name
        self.columns.append(col_name)

        """This code creates a new data column that is either correlated or not
         correlated with a given column in a provided data frame."""
        data, elapsed_time,correlation,found_p,numerical_data_frame,found_min   = create_data_from_column(
                                  self.numerical_data_frame,
                                  self.dataframe,
                                  columnname = "label",
                                  correlation = correlation, 
                                  corr_type = corr_type, 
                                  data_type = data_type,
                                  distribution = distribution,
                                  df = df,
                                  target_p = target_p,
                                  maxiteration=maxiteration,
                                  number_of_categories = number_of_categories,
                                  categorical_even = categorical_even
                                  )

        self.numerical_data_frame = numerical_data_frame
        
        header_param__temp_df                           = pd.DataFrame( index=(0,))  
        header_param__temp_df["label_index"]            = self.label_index
        header_param__temp_df["parameter_names"]        = col_name
        header_param__temp_df["distribution"]           = distribution

        header_param__temp_df["data_type"]              = data_type
        header_param__temp_df["corr_type"]              = corr_type
        header_param__temp_df["var_type"]               = var_type

        header_param__temp_df["maxiteration"]           = maxiteration
        header_param__temp_df["elapsed_time"]           = elapsed_time
        now = datetime.now()
        header_param__temp_df["current_time"]           = now.strftime("%H:%M:%S")

        #if data_type == "numerical":
   
        data_values                                     = std * data + expected_value
        data_values                                     = np.round(data_values,2)
        standard                                        = np.round(np.std(data_values))
        expected_value                                  = np.round(np.mean(data_values))

        
        header_param__temp_df["p"]                      = round(found_p,2)
        header_param__temp_df["correlation"]            = round(correlation,2)
        header_param__temp_df["std"]                    = std
        
        header_param__temp_df["categorical_even"]       = ''
        header_param__temp_df["found_min"]              = ''
        header_param__temp_df["number_of_categories"]   = ''

        print("created")
        print(" ")

        all_data_df = self.all_data_df

        data_values_df = pd.DataFrame(data_values , columns=[col_name])
        self.all_data_df = pd.concat([all_data_df,data_values_df],axis=1)
