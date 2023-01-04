#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 13:29:49 2020

@author: temuuleu
"""


import warnings
warnings.simplefilter("ignore", UserWarning)

import numpy as np
import h5py
import os
import json
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd
import missingno as mno
import seaborn as sns
from sklearn import linear_model
import math
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split



from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from bayes_opt import BayesianOptimization
from joblib import Parallel, delayed
from sklearn import svm, datasets
from matplotlib.pyplot import figure
from scipy.stats import ttest_ind




fig_with = 10
fig_height = 8
seed = 7


def check_missing_values(dataframe, plot = 0, show = 0, tolerance = 0, return_missing = 0):
    """Visualize the missing information of the dataframe
    
        
    Arguments:
        
        dataframe: DataFrame 
        plot:   int 
                    0: plot all , 
                    1: plot only missing image      
                    2: plto only not missing image
                    
        show:   int 
                    0: show all , 
                    1: show only missing image      
                    2: show only not missing image     
                    
                    
                    
        tolerance:  in percentage float,  
                    threshold to show only the columns that missing over the given 
                    percentage
                    
    Return:
        missing_columns :  return the missing columns
        not_missing_col :  return the rest of the columns
    
    """
    missing_col = []
    not_missing_col = []
    datalength = len(dataframe)

    for col in dataframe.columns:
        #if dataframe[col].isnull().sum():
        percentage_missing = dataframe[col].isnull().sum() / datalength 
        if percentage_missing >= tolerance:
            missing_col.append(col)
        else:
            not_missing_col.append(col)
            
            if show == 2 or show == 3:
                print("\nnot missing column \n")
                print("missing     : ", col)
                print("count       : ", dataframe[col].isnull().sum())
                print("percentage  : ", percentage_missing)

        if show == 1 or show == 3:
            print("\nmissing column \n")
            print("missing     : ", col)
            print("count       : ", dataframe[col].isnull().sum())
            print("percentage  : ", percentage_missing)
                
                
    print("column counts ", int(dataframe.shape[1]) )              
    print("tolerance at ", tolerance*100, " %" )                

    print("missing columns:", len(missing_col))    
    print("not missing columns:", len(not_missing_col))    
                
    if plot == 0 or plot == 3:

        mno.matrix(dataframe, figsize = (20,5))
        plt.title("Plot all columns " +str(dataframe.shape[1]))
        
    if plot == 1 and missing_col or plot == 3:

        mno.matrix(dataframe[missing_col], figsize = (20,5))
        plt.title("Plot missing columns " +str(len(missing_col)))
                
                
    if plot == 2 or plot == 3:     

        mno.matrix(dataframe[not_missing_col], figsize = (20,5))
        plt.title("Plot not missing columns " +str(len(not_missing_col)))
        

        
    if return_missing:    
        return missing_col,not_missing_col


def binary_encode_single_col(dataframe, colname, drop_multi_category =1):
    """cast a string into a binary int variable
    
    Arguments:
        dataframe: DataFrame to change
        colname:   String Column that has to change
        
    Return:
        the changed DataFrame
    
    """
    temp_df = dataframe.copy(deep=True)
    unique_names = temp_df.loc[:,colname].unique()
    is_binary_index = 0
    mapper = {}
    
    for index, name in enumerate(unique_names):
        if type(name)== str:
            if "nein" in name.lower() or "no" in  name.lower():
                mapper[name] = 0
            elif "ja" in name.lower() or "yes" in  name.lower():
                mapper[name] = 1
            else:
                mapper[name] = int(index) 
            is_binary_index+=1

        elif  math.isnan(name):
            mapper[name] = name

    if is_binary_index==2:
        temp_df[colname] = temp_df[colname].replace(mapper)
        return temp_df,mapper,""
    elif is_binary_index >2  or unique_names.dtype.name:
        if drop_multi_category:
            temp_df = temp_df.drop([colname], axis=1)
            return temp_df,  {} , colname
        else:
            temp_df = pd.get_dummies(temp_df, columns=[colname])

            return temp_df, {}, ""
    
    else:
        return temp_df, {} ,""
    
    
def hot_encoding(dataframe, drop_multi_category = 0, showmap  =0):
    """casts a string variabl into a binary integer variable
    
    Arguments:
        dataframe: DataFrame to change
        colname:   String Column that has to change
        
    Return:
        the changed DataFrame with a binary integer columns
    
    """
    temp_df = dataframe.copy(deep=True)
    
    droped_columns = []
    
    none_categorical_columsn = []

    maps = {   }
    for i, col in enumerate(temp_df.columns):
        if temp_df[col].dtypes.name=='category':
            temp_df, mapper, droped_colum = binary_encode_single_col(temp_df, col, drop_multi_category)
            if droped_colum:
                droped_columns.append(droped_colum)
            if mapper:
                maps.update( {col : mapper} )
        else:
            none_categorical_columsn.append(col)
        
    if showmap:       
        print("categorical col : ",droped_columns)
        print(maps)
            
    return temp_df,dataframe[droped_columns],maps,droped_columns,none_categorical_columsn


def random_imputation(df, feature):
    """
    
    """

    number_missing = df[feature].isnull().sum()
    observed_values = df.loc[df[feature].notnull(), feature]
    df.loc[df[feature].isnull(), feature + '_imp'] = np.random.choice(observed_values, number_missing, replace = True)
    
    return df

def impute_random(dataframe, missing_columns):
    """crete random values in the given dataframe"""
    #random imputation
    for feature in missing_columns:
        dataframe[feature + '_imp'] = dataframe[feature]
        dataframe = random_imputation(dataframe, feature)
    return dataframe


def create_det(dataframe, missing_columns, text = "Det", method = "linear"):
    """Inpute the missing columns with liniar regression method
    
    ?: should there be other interpolation method other than linear regression?
    
    Arguments:
        dataframe: DataFrame to change
        
        missing_columns:   List of columnnames that is going to be inputed
        
    Return:
        dataframe: DataFrame with inputed data
    
    """
    temp_df = dataframe.copy(deep=True)
    #input randomly
    imputed_df = impute_random(temp_df, missing_columns)
    #create dataset with changed colname
    deter_data = pd.DataFrame(columns = [text + name for name in missing_columns])
    
    for feature in missing_columns:

        deter_data[text + feature] = imputed_df[feature + "_imp"]
        parameters = list(set(imputed_df.columns) - set(missing_columns) - {feature + '_imp'})

        #Create a Linear Regression model to estimate the missing data
        if method=="linear":
            model = linear_model.LinearRegression()
            model.fit(X = imputed_df[parameters], y = imputed_df[feature + '_imp'])

        #observe that I preserve the index of the missing data from the original dataframe
        deter_data.loc[imputed_df[feature].isnull(), text + feature] = model.predict(imputed_df[parameters])[dataframe[feature].isnull()]

    deter_data = deter_data[[colname for colname in deter_data.columns if text in colname]]  
    
    rename_dict = {}
    for colname in deter_data.columns:
        rename_dict[colname] = colname[len(text):]
    
    df_rename = deter_data.rename(columns=rename_dict)
    
    
    not_missing_colsumns = list_minus_list(dataframe.columns, missing_columns)

    not_missing_df = dataframe[not_missing_colsumns]

    full_df = pd.concat([not_missing_df, df_rename], axis= 1)
    
    return full_df

def list_minus_list(list1, list2):
    return [c for c in list1 if c not in list2]

