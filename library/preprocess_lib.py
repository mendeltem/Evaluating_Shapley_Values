#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 13:51:23 2021

@author: temuuleu
"""
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter(action='ignore', category=Warning)
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from copy import copy as cp

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix

#import shap
from imblearn.over_sampling import SMOTE
from collections import Counter

import math

def ratio(p1,p2):
    ##get the ratio of 2 numercial values
    
    if p1 >= p2:
        smaller = p2
        bigger  = p1
    else:
        smaller = p1
        bigger  = p2
        
    return smaller / bigger  

def undummify(df, prefix_sep="_"):
    cols2collapse = {
        item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
    }
    series_list = []
    for col, needs_to_collapse in cols2collapse.items():
        if needs_to_collapse:
            undummified = (
                df.filter(like=col)
                .idxmax(axis=1)
                .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                .rename(col)
            )
            series_list.append(undummified)
        else:
            series_list.append(df[col])
    undummified_df = pd.concat(series_list, axis=1)
    return undummified_df


def create_data(cleaned_data_df, in_dict,out_dict, list_of_types ):
    
    labels         = list(out_dict.values())
    
    Data = {
        "in_dict"                   : in_dict,
        "out_dict"                  : out_dict,
        "labels"                    : list(out_dict.values()),
        "list_of_types"             : list(list_of_types)
    }
    
    for label in labels:
        print(label)
        Data[label] = {}
        Data[label]['y']  = cleaned_data_df[label].dropna()
        
        print(len(Data[label]['y']))
    
        Data[label]["Input_datasets"] = {}
        for feature_type in list_of_types:
            X = cleaned_data_df.loc[Data[label]['y'].index,list_of_types[feature_type]]
        
            Data[label]["Input_datasets"][feature_type] = {}
            Data[label]["Input_datasets"][feature_type]["X"] = X.loc[:,list_of_types[feature_type]]
            category_map,categorical_features,ordinal_features, feature_names = create_category_map(X.loc[:,list_of_types[feature_type]].drop(columns = ["sex_index"]))
            Data[label]["Input_datasets"][feature_type]["category_map"]         = category_map
            Data[label]["Input_datasets"][feature_type]["categorical_features"] = categorical_features
            Data[label]["Input_datasets"][feature_type]["ordinal_features"]     = ordinal_features
            Data[label]["Input_datasets"][feature_type]["feature_names"]        = feature_names
            
    return Data


def get_sampe(df,size, start, end, sex ,seed = 0):
    """get sample of data given bins from start to end of sex
    
    Arguments:
        df:     DataFrame to change
        size:   count to sample
        start:  start index
        end:    end index
        sex:    gender
        
    Return:
        pandas series

    """
    
    import random
    
    random.seed(seed)
    
    sample_list= (df["sex"] == sex) & (df["age_y"] >= start) & (df["age_y"] < end)
    indizes = [index for index,r in enumerate(sample_list) if r == True]

    random_index = sample_list[random.sample(indizes, k=size)]
    random_index = random_index.sort_index()
    series = []
    for i in range(0,df.shape[0]):

        if i in random_index:
            series.append(True) 
        else:
            series.append(False) 

    return pd.Series(series)


def create_category_map(X_data , limit = 10):
    """creates category map from a DataFrame and lists of columnnumbers
    
    
    
    input : DataFrame
            limit: shows when the variable is a float or a category
                    every variable with over 10 values are floats
            
    return: Dictionary  :  categorymap 
            List        :  column number of category columns
            List        :  column number of numerical columns
    """
    
    category_map = {}
    
    ordinal_features = []
    categorical_features = []

    for i,col_name in enumerate(X_data.columns):
        unique_len = len(X_data[col_name].dropna().unique())
        
        if unique_len == 2 or unique_len < limit:
            categorical_features.append(i)
            
            category_map[i] = {}
            category_map[i] = []
            cat_values = list(X_data[col_name].unique())

            for cat_value in cat_values:
                
                category_map[i].append(cat_value)
        else:
            ordinal_features.append(i)
                        

    return category_map,categorical_features,ordinal_features,list(X_data.columns)


# def preprocess(train_index, test_index):
#     """
#     Imputation and smote
    
    
#     """
    
    
#     X_train, X_test = X_data.iloc[train_index, :], X_data.iloc[test_index, :]
    
#     #train categorical imputeter on train set
#     if categorical_features:
#         cat_imp.fit(X_train.iloc[:,categorical_features]) 
    
#         #impute categorical training set
#         X_train_categorical_imputed       = pd.DataFrame(cat_imp.transform(X_train[categorical_features_names]),
#                                                     columns = list(X_train[categorical_features_names].columns))

#         X_train_categorical_imputed.index = X_train[categorical_features_names].index

#         #impute categorical test set
#         X_test_categorical_imputed        = pd.DataFrame(cat_imp.transform(X_test[categorical_features_names]), 
#                                                           columns = X_train[categorical_features_names].columns)

#         X_test_categorical_imputed.index = X_test[categorical_features_names].index
    

    
#         X_train_categorical_numerical    = pd.concat([X_train[numerical_feaures_names],
#                                                         X_train_categorical_imputed],axis=1)


#         X_test_categorical_numerical     = pd.concat([X_test[numerical_feaures_names],
#                                                       X_test_categorical_imputed],axis=1)
#     else:
#         X_train_categorical_numerical    = X_train
#         X_test_categorical_numerical     = X_test
        
    
#     #numerical imputation
#     #train iterat imputer on full train set
#     iter_imp.fit(X_train_categorical_numerical)
#     #train simple imputer on full train set
#     num_imp.fit(X_train_categorical_numerical)
    
#     #impute iterative from training
#     X_Train_imputed  = iter_imp.transform(X_train_categorical_numerical)
#     X_train_df       = pd.DataFrame(X_Train_imputed, columns = X_train_categorical_numerical.columns)
#     X_train_df.index = X_train_categorical_numerical.index

#     #impute mean from training
#     X_Test_imputed   = num_imp.transform(X_test_categorical_numerical)
#     X_test_df        = pd.DataFrame(X_Test_imputed, columns   = X_test_categorical_numerical.columns)
#     X_test_df.index  = X_test_categorical_numerical.index
    
    
#     #one hot encoding
#     #concatenate train and test for one hot encoding
#     full_imputed = pd.concat((X_train_df, X_test_df))

#     full_dummies = pd.get_dummies(full_imputed, columns = multi_categorical_features,prefix_sep="+")
#     full_dummies = full_dummies.sort_index()
    
                    
#     #rename feature columns for one hot encoded
#     hot_encoded_feature_names = []
#     hot_encoded_feature_names.append("sex_index")

#     for feature_name in feature_names:
#         for new_col in list(full_dummies.columns):
#             if "sex_index" not in new_col:
#                 if feature_name in new_col:
#                     hot_encoded_feature_names.append(new_col) 
                    
#     #train test split with imputed dataset
#     y_train, y_test = pd.DataFrame(Y_data).iloc[train_index,0], pd.DataFrame(Y_data).iloc[test_index,0]   
#     x_train, x_test = full_dummies.iloc[train_index,:],full_dummies.iloc[test_index,:]


#     #smote for unbalanced labels
#     X_ratio = Counter(y_train)
#     porportion = ratio(X_ratio[1], X_ratio[0])
    
    
#     imputed_data = {}

#     if porportion < 0.8:
#         X_smot, y_smot = smote.fit_resample(x_train,y_train)

#         for col in categorical_features_names:
#             for x_col in X_smot.columns:
#                 if col in x_col:
#                     X_smot[x_col] = cp(round(X_smot[x_col]))

#         for x_col in X_smot.columns:
#             X_smot[x_col] = round(X_smot[x_col],2) 
        
#         imputed_data["train_X"]  = X_smot
#         imputed_data["train_y"]  = y_smot
#     else:
#         imputed_data["train_X"]  = x_train
#         imputed_data["train_y"]  = y_train
        
    
#     imputed_data["test_X"]  = x_test
#     imputed_data["test_y"]  = y_test
        
    
#     return imputed_data

def preprocess(train_index, 
               test_index,
               X_data,
               Y_data,
               categorical_features,
               cat_imp,
               iter_imp,
               num_imp,
               categorical_features_names,
               numerical_feaures_names,
               multi_categorical_features,
               feature_names,
               seed
               ):
    """
    This function allows to split the dataset in to train and test.
    Also this preprocessing function imputes the data, synthetize
    imbalanced output with smote and one hot encode
    variable with multiple categories

    
    Arguments:
        train_index:                list
        test_index:                 list
        X_data:                     DataFrame     Input Values
        Y_data:                     Pandas Series Outcome Variables
        categorical_features:       list 
        cat_imp:                    simple imputer for categories
        iter_imp:                   imputer for numbers
        num_imp:                    simple imputer for numbers
        categorical_features_names: list
        numerical_feaures_names:    list
        multi_categorical_features: list
        feature_names:              list
        
    Return:
        Imputed data:    DataFrame

    """
    
    
    
    X_train, X_test = X_data.iloc[train_index, :], X_data.iloc[test_index, :]
    
    
    smote = SMOTE(0.9, random_state=seed, n_jobs=-1)
    
    #train categorical imputeter on train set
    if categorical_features:
        cat_imp.fit(X_train.iloc[:,categorical_features]) 
    
        #impute categorical training set
        X_train_categorical_imputed       = pd.DataFrame(cat_imp.transform(X_train[categorical_features_names]),
                                                   columns = list(X_train[categorical_features_names].columns))

        X_train_categorical_imputed.index = X_train[categorical_features_names].index

        #impute categorical test set
        X_test_categorical_imputed        = pd.DataFrame(cat_imp.transform(X_test[categorical_features_names]), 
                                                         columns = X_train[categorical_features_names].columns)

        X_test_categorical_imputed.index = X_test[categorical_features_names].index
    

    
        X_train_categorical_numerical    = pd.concat([X_train[numerical_feaures_names],
                                                       X_train_categorical_imputed],axis=1)


        X_test_categorical_numerical     = pd.concat([X_test[numerical_feaures_names],
                                                      X_test_categorical_imputed],axis=1)
    else:
        
        X_train_categorical_numerical    = X_train
        
        X_test_categorical_numerical     = X_test
        

    #numerical imputation
    #train iterat imputer on full train set
    iter_imp.fit(X_train_categorical_numerical)
    #train simple imputer on full train set
    num_imp.fit(X_train_categorical_numerical)
    
    #impute iterative from training
    X_Train_imputed  = iter_imp.transform(X_train_categorical_numerical)
    X_train_df       = pd.DataFrame(X_Train_imputed, columns = X_train_categorical_numerical.columns)
    X_train_df.index = X_train_categorical_numerical.index

    #impute mean from training
    X_Test_imputed   = num_imp.transform(X_test_categorical_numerical)
    X_test_df        = pd.DataFrame(X_Test_imputed, columns   = X_test_categorical_numerical.columns)
    X_test_df.index  = X_test_categorical_numerical.index
    
    
    #one hot encoding
    #concatenate train and test for one hot encoding
    full_imputed = pd.concat((X_train_df, X_test_df))

    full_dummies = pd.get_dummies(full_imputed, columns = multi_categorical_features, prefix_sep='+')
    full_dummies = full_dummies.sort_index()
    
                    
    #rename feature columns for one hot encoded
    hot_encoded_feature_names = []
    hot_encoded_feature_names.append("sex_index")

    for feature_name in feature_names:
        for new_col in list(full_dummies.columns):
            if "sex_index" not in new_col:
                if feature_name in new_col:
                    hot_encoded_feature_names.append(new_col) 
                    
    #train test split with imputed dataset
    #one hot encode with fulldummies
    y_train, y_test = pd.DataFrame(Y_data).iloc[train_index,0], pd.DataFrame(Y_data).iloc[test_index,0]   
    x_train, x_test = full_dummies.iloc[train_index,:],full_dummies.iloc[test_index,:]

    #smote for unbalanced labels
    X_ratio = Counter(y_train)
    porportion = ratio(X_ratio[1], X_ratio[0])
    
    imputed_data = {}

    #if the ratio between output values 1 and 0 are bigger the 0.8 then use smote
    if porportion < 0.8:
        X_smot, y_smot = smote.fit_resample(x_train,y_train)

        for col in categorical_features_names:
            for x_col in X_smot.columns:
                if col in x_col:
                    X_smot[x_col] = cp(round(X_smot[x_col]))

        for x_col in X_smot.columns:
            X_smot[x_col] = round(X_smot[x_col],2) 
        
        imputed_data["train_X"]  = round_columns(X_smot)
        imputed_data["train_y"]  = y_smot
    else:
        imputed_data["train_X"]  = round_columns(x_train)
        imputed_data["train_y"]  = y_train
        
    
    imputed_data["test_X"]  = round_columns(x_test)
    imputed_data["test_y"]  = y_test
        
    
    return imputed_data



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
    
    
    
def list_minus_list(list1, list2):
    return [c for c in list1 if c not in list2]


def append_string_length(text,lenght = 40, char=" "):
    """the string lenght get appended to have the given length"""

    diff = lenght - len(text)
    for u in range(diff):
        text += char
    return text


def how_many_decimals(number):
    """check if the number has decimals"""
    return number - int(number) == 0

def round_columns(X, percentile = 0.5, med = 50):
    
    for col in X.columns:

        percent_decimal = round(len([val for val  in X[col] if not how_many_decimals(val)])/ len(X[col]),2)

        if percent_decimal < percentile or X[col].median() > med:
            X[col] = round(X[col])
    
    return X


def ratio(p1,p2):
    ##get the ratio of 2 numercial values
    
    if p1 >= p2:
        smaller = p2
        bigger  = p1
    else:
        smaller = p1
        bigger  = p2
        
    return smaller / bigger  



def create_dir(output_path):
    """creates a directory of the given path"""
    if not os.path.exists(output_path):
        os.makedirs(output_path)
