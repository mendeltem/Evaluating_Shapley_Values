#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 21:50:24 2021

@author: temuuleu
"""
import numpy as np
import os
import re
from contextlib import contextmanager
import shap
import pandas as pd
import pickle
import numpy as np
from copy import copy as cp
import matplotlib.pyplot as plt

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
from scipy.special import logit
#import shap
from imblearn.over_sampling import SMOTE
from collections import Counter
from collections import OrderedDict
import random
from timeit import default_timer as timer
import math
from joblib import Parallel, delayed
import os
import sys, os



def undummify(df, prefix_sep="+"):
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


def recreate(path):
    
   create_dir(f"{path}")
   
   if os.path.exists(path):
       os.system(f"rm -r {path}")
   
   create_dir(f"{path}")

def collect_hot_encoded_features_kernel(data, shap_values, seperator="+"):
    """creates category map from a DataFrame and lists of columnnumbers
    
    
    
    input : DataFrame  : Data
            shap_values: shows when the variable is a float or a category
                    every variable with over 10 values are floats
            
    return: Dictionary  :  categorymap 
            List        :  column number of category columns
            List        :  column number of numerical columns
    """
    
    dictionary = {}
    new_shap_values = []
    
    col_counter = 0
    col_bool    = 0
    
    for col_idx,col in enumerate(data.columns):
        if seperator in  col:
            new_col = col[:-4]
            if new_col not in dictionary.keys():
                if col_bool == 0:
                    col_counter = col_idx
                    col_bool  =1
                else:
                    col_counter += 1
                
                dictionary[new_col] = {}
                dictionary[new_col][new_col+"_old"] = []
                
                dictionary[new_col][new_col+"_old"].append(col_idx)
                
                dictionary[new_col][new_col] = col_counter
            else:
                dictionary[new_col][new_col+"_old"].append(col_idx)
        else:
            dictionary[col] = {}
            dictionary[col][col] = col_idx
            
    for shap_value_part in shap_values:
        array = np.zeros((shap_value_part.shape[0],len(dictionary)))

        for id_x in range(shap_value_part.shape[0]):

            for i,k in enumerate(dictionary.keys()):
                if len(dictionary[k]) > 1:
                    #array[id_x][dictionary[k][k]] = sum(shap_value_part[id_x][dictionary[k][k+"_old"]])/ len(shap_value_part[id_x][dictionary[k][k+"_old"]])
                    array[id_x][dictionary[k][k]] = sum(shap_value_part[id_x][dictionary[k][k+"_old"]])
                else:
                    array[id_x][dictionary[k][k]] = shap_value_part[id_x][dictionary[k][k]]   

        new_shap_values.append(array)
                
    return new_shap_values, list(dictionary.keys())


def collect_hot_encoded_features(data, shap_values, seperator="+"):
    """creates category map from a DataFrame and lists of columnnumbers
    
    
    
    input : DataFrame  : Data
            shap_values: shows when the variable is a float or a category
                    every variable with over 10 values are floats
            
    return: Dictionary  :  categorymap 
            List        :  column number of category columns
            List        :  column number of numerical columns
    """
    dictionary = {}
    new_shap_values = []
    
    col_counter = 0
    col_bool    = 0
    
    for col_idx,col in enumerate(data.columns):
        if seperator in  col:
            new_col = col[:-4]
            if new_col not in dictionary.keys():
                if col_bool == 0:
                    col_counter = col_idx
                    col_bool  =1
                else:
                    col_counter += 1
                
                dictionary[new_col] = {}
                dictionary[new_col][new_col+"_old"] = []
                
                dictionary[new_col][new_col+"_old"].append(col_idx)
                
                dictionary[new_col][new_col] = col_counter
            else:
                dictionary[new_col][new_col+"_old"].append(col_idx)
        else:
            dictionary[col] = {}
            dictionary[col][col] = col_idx
            

    array = np.zeros((shap_values.shape[0],len(dictionary)))

    for id_x in range(array.shape[0]):

        for i,k in enumerate(dictionary.keys()):
            if len(dictionary[k]) > 1:
                array[id_x][dictionary[k][k]] = sum(shap_values[id_x][dictionary[k][k+"_old"]])
            else:
                array[id_x][dictionary[k][k]] = shap_values[id_x][dictionary[k][k]]   
    
    return array, list(dictionary.keys())



def plot_foreplot(data_test_x,explainer, data_type_str,model_name,plot_path, seperator = "+",contribution_threshold = 0,dpi=100):

    shap_values_test = explainer.shap_values(shap.sample(data_test_x))
    
    temp_shap_values, feature_names = collect_hot_encoded_features_kernel(data_test_x,shap_values_test)                                                                 
    data_test_x_undummied = undummify(data_test_x, prefix_sep=seperator)
    
    
    for f in range(temp_shap_values[0].shape[0]):
        temp_shap_values_test_tp = temp_shap_values[0][f]*-1
        temp_input_values = data_test_x_undummied.round(2).iloc[f]
        temp_exp_values = explainer.expected_value[0]
        temp_feature_names = data_test_x_undummied.columns
        
        short_name = shorten_names(temp_feature_names)
        temp_feature_names = short_name
        
        shap.force_plot(temp_exp_values,
                        temp_shap_values_test_tp,
                        temp_input_values,
                        temp_feature_names,
                        contribution_threshold=contribution_threshold,
                        plot_cmap = ["#ff4d4d", "#DCDCDC"],
                        matplotlib=True,                                        
                        show=False)
        
        shap_plot_path =  os.path.join(plot_path,data_type_str)
        create_dir(shap_plot_path)
        kernel_shap_plot_path =  os.path.join(shap_plot_path,model_name+f"_kernel_shap_fore_{f}_{data_type_str}.png")
        plt.savefig(kernel_shap_plot_path, 
                         dpi=dpi,
                        bbox_inches ="tight",
                         pad_inches=1)
        plt.clf() 
        

def shorten_names(temp_feature_names):

    short_names  = []
    for name in list(temp_feature_names):
        
        short_name  = []
        for n in name.split("_"):
            print(n)
            print(n.isdigit())
            print("")
            if not n.isdigit():
                short_name+= [n]
                
        short_name = list(set(short_name))
        short_names += ["_".join(short_name)]
        
    return short_names

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


def save_training_data(full_data,parameter_names_list, test_case_factor_path  , head_df , dpi = 100 , label_name = ""):

    X = full_data.loc[:,parameter_names_list]
    y = full_data.loc[:,"label" ]
    
    DATA =  pd.concat([y,X], axis = 1)
    create_dir(test_case_factor_path)
    
    y.hist(color="red")
    plt.title("label")
    plt.savefig(os.path.join(test_case_factor_path,"label.png"), 
                dpi=dpi,
                bbox_inches ="tight",
                pad_inches=1)
                     
    plt.clf() 
    
    for parameter_name in parameter_names_list:
        column_data = full_data.loc[:,parameter_name]
        column_data.hist()
        plt.title(f"{parameter_name}")
        plt.savefig(os.path.join(test_case_factor_path,parameter_name+".png"), 
                    dpi=dpi,
                    bbox_inches ="tight",
                    pad_inches=1)
                         
        plt.clf() 

    input_path = os.path.join(test_case_factor_path,"DATA.xlsx")
    DATA.to_excel(input_path,index=False)
    
    corr = DATA.corr()
    corr_path = os.path.join(test_case_factor_path,"cor.xlsx")
    corr.to_excel(corr_path,index=False)
    
    header_path = os.path.join(test_case_factor_path,"header.xlsx")
    head_df.to_excel(header_path,index=False)
    
    
def save_header(parameter_name, head_df , meta_data_label , parameter_names_list ):
    
    parameter_names_list+=[parameter_name]
   
    temp_head_df = pd.DataFrame(index=(parameter_name,))
    temp_head_df["df"]                    = meta_data_label.loc[meta_data_label["parameter_names"] == parameter_name,["df"]].values[0][0]
    temp_head_df["label_ratio"]           = meta_data_label.loc[meta_data_label["parameter_names"] == parameter_name,["label_ratio"]].values[0][0]
    temp_head_df["correlation"]           = meta_data_label.loc[meta_data_label["parameter_names"] == parameter_name,["correlation"]].values[0][0]
    temp_head_df["corr_type"]             = meta_data_label.loc[meta_data_label["parameter_names"] == parameter_name,["corr_type"]].values[0][0]
    temp_head_df["std"]                   = meta_data_label.loc[meta_data_label["parameter_names"] == parameter_name,["std"]].values[0][0]
    temp_head_df["p"]                     = meta_data_label.loc[meta_data_label["parameter_names"] == parameter_name,["p"]].values[0][0]
    temp_head_df["number_of_categories"]  = meta_data_label.loc[meta_data_label["parameter_names"] == parameter_name,["number_of_categories"]].values[0][0]
    temp_head_df["data_type"]             = meta_data_label.loc[meta_data_label["parameter_names"] == parameter_name,["data_type"]].values[0][0]
    temp_head_df["label_ratio"]           = meta_data_label.loc[meta_data_label["parameter_names"] == parameter_name,["label_ratio"]].values[0][0]
    temp_head_df["distribution"]          = meta_data_label.loc[meta_data_label["parameter_names"] == parameter_name,["distribution"]].values[0][0]
    temp_head_df["var_type"]              = meta_data_label.loc[meta_data_label["parameter_names"] == parameter_name,["var_type"]].values[0][0]
    temp_head_df["categorical_even"]              = meta_data_label.loc[meta_data_label["parameter_names"] == parameter_name,["categorical_even"]].values[0][0]
    
    temp_head_df["parameter_name"]        = parameter_name

    head_df = pd.concat([head_df,temp_head_df])
    
    return head_df,parameter_names_list

def check_nan_in_df(X):
    
    if type(X) == pd.core.series.Series :
        nan = X[X.isna()].shape[0]
        
    elif type(X) == pd.core.frame.DataFrame:
        nan = X[X.isna().any(axis=1)].shape[0]
        
    if nan:
        print(nan)


def combine_paths(paths=[]):
    """
    combin paths together
    
    """
    
    
    combined_pat = ""

    for path in paths:
        combined_pat+= path + "/"

    
    if "//" in combined_pat:
        combined_pat = combined_pat.replace('//', '/')
    
    if "///" in combined_pat:
        combined_pat = combined_pat.replace('///', '//') 

    if "////" in combined_pat:
        combined_pat = combined_pat.replace('////', '//')     
        
        
    while '/' == combined_pat[-1]:
        combined_pat = combined_pat[:-1]
        
        
    return combined_pat

def list_files(path):


    return  [combine_paths([path,directory])\
                 for directory in os.listdir(path)]

        
def is_file(path_name):
    """check if the given string is a file"""
    if re.search("\.[a-zA-Z]+$", os.path.basename(path_name)):
        return True
    else:
        return False

def is_directory(path_name):
    
    #path_name = "/persDaten/MRT_daten_manual/output."
    """check if the given string is a directory"""
    
    ewp = os.path.basename(path_name).endswith('.')

    if not ewp and not is_file(path_name) and not len(os.path.basename(path_name))  == 0:
        return True
    else:
        return False

def create_dir(output_path):
    """creates a directory of the given path"""
    if not os.path.exists(output_path):
        os.makedirs(output_path)

def how_many_decimals(number):

    """check if the number has decimals"""

    return number - int(number) == 0


def round_columns(X, percentile = 0.5, med = 50):

    for col in X.columns:

        percent_decimal = round(len([val for val  in X[col] if not how_many_decimals(val)])/ len(X[col]),2)

        if percent_decimal < percentile or X[col].median() > med:

            X[col] = round(X[col])

    return X


def _parse_results(cv_results, train_scores=False):

    dic_results = OrderedDict()
    # Add mean score

    best_idx = cv_results["rank_test_score"].argmin()
    dic = OrderedDict({"score_mean_test": cv_results["mean_test_score"][best_idx]})

    if train_scores:
        dic.update({"score_mean_train": cv_results["mean_train_score"][best_idx]})

    best_params = cv_results['params'][best_idx]

    # add values of the highest ranked parameters

    for par, val in best_params.items():
        dic[par] = val

    dic_results.update(dic)

    return dic_results


def get_ratio(x,y):

    cy = int(round(y / (math.gcd(x, y)),0))
    cx = int(round(x / (math.gcd(x, y)),0))
    
    if cx > cy:
        label_ratio                         = round( cy/cx,2)
    else:
        label_ratio                         = round( cx/cy,2)
    
    return label_ratio


def create_labels(number_of_data = 1000, ratio = 1 , max_iter = 100):
        
    tmp_dict = {}

    def preprocess(i,number_of_data, prob):

        tmp_dict = {}
        tmp_dict[i] = {}

        labels                              =  np.random.choice([0, 1], 
                                                  size=number_of_data, 
                                                  p=[round(1 -prob,2) ,prob])
        
        label_ratio                         = get_ratio(Counter(labels)[1], Counter(labels)[0])
        diff_ratio                          = abs(ratio - label_ratio)

        tmp_dict[i]["label_ratio"]          = label_ratio
        tmp_dict[i]["diff_ratio"]           = diff_ratio
        tmp_dict[i]["labels"]                = labels
        
        
        return tmp_dict

    result = pd.DataFrame()
    
    new_tmp_dict = {}
    index_p = 0
    
    
    list_num = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    np.random.shuffle(list_num)

    for prob in list_num:
        print(prob)
        
        tmp_dict=Parallel(n_jobs=-1, verbose=1, pre_dispatch='1.5*n_jobs')(
            delayed(preprocess)(i,number_of_data, prob) for i in range(max_iter))

        
        for i_u,u in enumerate(tmp_dict):
                   
            tmp_df = pd.DataFrame(index = (index_p,))

            
            label_ratio   = u[i_u]['label_ratio']
            diff_ratio    = u[i_u]['diff_ratio']
            labels        = u[i_u]['labels']
            
 
            tmp_df["i"] = index_p
            tmp_df["label_ratio"]  = label_ratio
            tmp_df["diff_ratio"]   = diff_ratio
            
            if diff_ratio == 0:  break
            
            #tmp_df["labels"]       = labels
                        
            new_tmp_dict[index_p]  = {}
            new_tmp_dict[index_p]["label_ratio"]  = label_ratio
            new_tmp_dict[index_p]["labels"]  = labels
            
            result  = pd.concat([result, tmp_df])
            
            index_p +=1
            
  
    result.sort_values(by=["diff_ratio"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    found_index =  result.loc[0,"i"]
    
    labels=new_tmp_dict[found_index]['labels']
    label_ratio=new_tmp_dict[found_index]['label_ratio']
    
    return labels, label_ratio


# def create_sum_of_probs(number_of_categories = 3, max_iteration =1000, categorical_even = False):
    

#     def preprocess(i,number_of_categories):
        
#         tmp_dict = {}

#         lam = lambda x: random.choice( [round(p*0.1,1) for p in range(1,10,1)])
        
#         ps =([lam(0) for l in  range(number_of_categories) ])

#         if sum(ps) == 1 : 
#             tmp_dict[i] = {}
#             tmp_dict[i]["ps"] = ps
#             tmp_dict[i]["sum"] = sum(ps)

#         return tmp_dict
    
#     tmp_dict=Parallel(n_jobs=-1, verbose=1, pre_dispatch='1.5*n_jobs')(
#         delayed(preprocess)(i,number_of_categories) for i in range(max_iteration))

#     result = pd.DataFrame()
#     pro_dict = {}
#     pi = 0
#     for i,t in enumerate(tmp_dict[:]):
        
#         if t:
#             pi+=1
#             pro_dict[pi] = {}
#             pro_dict[pi]["ps"] =  t[i]["ps"]
#             pro_dict[pi]["min"] =  sorted(t[i]["ps"])[0]
            
#             tmp_df = pd.DataFrame(index = (pi,))
#             tmp_df["i"] = pi
#             tmp_df["min"] = sorted( t[i]["ps"])[0]

#             result  = pd.concat([result, tmp_df])
            

#     result.sort_values(by=["min"], inplace=True, ascending=not(categorical_even))
#     result.reset_index(drop=True, inplace=True)
#     found_index =  result.loc[0,"i"]

#     target_generated_data=pro_dict[found_index]['ps']
#     found_min=0

#     return target_generated_data , found_min



def create_sum_of_probs(number_of_categories = 3, max_iteration =1000, categorical_even = False):
    

    lam = lambda x: random.choice( [round(p*0.1,1) for p in range(1,10,1)])
    
    for i in range(max_iteration):
        
        ps =([lam(0) for l in  range(number_of_categories) ])
        
        if sum(ps) == 1 : 
            
            target_generated_data = ps
            break

    return target_generated_data , sorted(ps)[0]



def create_random_categorical(labels, number_of_categories = 2, categorical_even = True):

    number_of_labels = len(labels)

    probabilities , found_min       = create_sum_of_probs(number_of_categories,max_iteration =5000,
                                                          categorical_even= categorical_even)

    categories                       = [i for i in range(1,number_of_categories+1)]
    
    new_categorical_data             = np.round(np.random.choice(categories, 
                                            size=number_of_labels, p=probabilities))
    
    return new_categorical_data, found_min,probabilities


     
def create_random_categorical_with_p(labels,number_of_categories,
                                     target_p = 0.05,
                                     maxiteration = 1000 ,
                                     categorical_even = True):
    
    def preprocess_1(i,labels,number_of_categories):
        

    
        tmp_dict = {}
        generated_data, found_min,probabilities=create_random_categorical(labels,
                                  number_of_categories, categorical_even)
        
        
        chis2,p = get_p_chi(generated_data,labels)
        p = round(p,4)

        tmp_dict[i] = {}
        tmp_dict[i]["p"] = p
        tmp_dict[i]["chis2"] = chis2
        tmp_dict[i]["generated_data"] = generated_data
        tmp_dict[i]["found_min"] = found_min
        
        return tmp_dict
    
    
    try_counter = 10

    while(try_counter):
    
        try:
            tmp_dict=Parallel(n_jobs=-1, verbose=1, pre_dispatch='1.5*n_jobs')(
                delayed(preprocess_1)(i,labels,number_of_categories) for i in range(maxiteration))
            
            break
        except:
            print(f"preprocess_1 .. try again   {try_counter}")
            
            try_counter -=1
            pass
    
    
    result = pd.DataFrame()
    
    new_tmp_dict = {}
    
    for i, d in enumerate(tmp_dict):
        
        index_ =list(d[i].keys())[0]

        p_val               = d[i]['p']
        generated_data      = d[i]['generated_data']
        
        chis2               = d[i]['chis2']
        found_min           = d[i]['found_min']
        
        new_tmp_dict[i] ={}
        
        new_tmp_dict[i]['generated_data'] = generated_data
        new_tmp_dict[i]['p']              = p_val
        new_tmp_dict[i]['chis2']          = chis2
        new_tmp_dict[i]['found_min']      = found_min
        
        tmp_df = pd.DataFrame(index = (i,))
        tmp_df["i"]                = i
        tmp_df["p"]                = p_val
        tmp_df["chis2"]            = chis2
        tmp_df["found_min"]        = found_min
        result                     = pd.concat([result, tmp_df])

    result["target"]  = abs(result["p"] - target_p  )
    result.sort_values(by=["target","found_min"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    found_index =  result.loc[0,"i"]
    
    target_generated_data=new_tmp_dict[found_index]['generated_data']
    found_chis2=new_tmp_dict[found_index]['chis2']
    found_p=new_tmp_dict[found_index]['p']
    found_min=new_tmp_dict[found_index]['found_min']
    
    
    return target_generated_data, found_chis2,found_p,found_min





def get_p_chi(new_categorical_data,labels):
    
    CrosstabResult=pd.crosstab(index=new_categorical_data,columns=labels)
    #print(CrosstabResult)
     
    # importing the required function
    from scipy.stats import chi2_contingency
     
    # Performing Chi-sq test
    ChiSqResult = chi2_contingency(CrosstabResult)
    return ChiSqResult[0],ChiSqResult[1]


def create_cross_p_values(part_categorical_work):

    df_corr = pd.DataFrame()
    
    for c_c,c in enumerate(part_categorical_work.columns):
        
        data_fame = pd.DataFrame(index=(c,)) 
        
        for r_i,r in  enumerate(part_categorical_work.columns):
            chis2,p_value = get_p_chi(part_categorical_work[c],part_categorical_work[r])
            data_fame[r] = round(p_value,2)
            
        df_corr  = pd.concat([df_corr , data_fame])
        
    return df_corr
        

def create_dir(output_path):
    """creates a directory of the given path"""
    if not os.path.exists(output_path):
        os.makedirs(output_path)



def force_plot_true(X_test_ci, explainer, y_test_ci, y_test_pred ,model_name,plot_path):

    #true positiv
    tp_i_test = [i for i,y in enumerate(y_test_ci) if y == 1 and y_test_pred[i] == 1]
    
    if tp_i_test:
        data_test_x = X_test_ci.iloc[tp_i_test[:5]]
        data_type_str = "tp"
        plot_foreplot(data_test_x,explainer, data_type_str,model_name,plot_path)
    
    #true negativ
    tn_i_test = [i for i,y in enumerate(y_test_ci) if y == 0 and y_test_pred[i] == 0]
    if tn_i_test:
        data_test_x = X_test_ci.iloc[tn_i_test[:5]]
        data_type_str = "tn"
        plot_foreplot(data_test_x,explainer, data_type_str,model_name,plot_path)
    
    
    #false positiv
    fp_i_test = [i for i,y in enumerate(y_test_ci) if y == 0 and y_test_pred[i] == 1]
    if fp_i_test:
        data_test_x = X_test_ci.iloc[fp_i_test[:5]]
        data_type_str = "fp"
        plot_foreplot(data_test_x,explainer, data_type_str,model_name,plot_path)
        

    #false neg
    fp_i_test = [i for i,y in enumerate(y_test_ci) if y == 1 and y_test_pred[i] == 0]
    if fp_i_test:
        data_test_x = X_test_ci.iloc[fp_i_test[:5]]
        data_type_str = "fn"
        plot_foreplot(data_test_x,explainer, data_type_str,model_name,plot_path)

def delet_nan_from_list(given_list, if_set = True):
    
    if if_set:
        return  list(set([u for u in given_list if not "nan" == str(u).lower() ]))
    else:
        return  [u for u in given_list if not "nan" == str(u).lower() ]

from scipy.stats import pearsonr
from scipy.optimize import fmin, minimize



def create_random_numerical_data(column,number_of_data,numerical_data_frame,distribution,maxiteration= 1000,df =1):
    
    n_length                          = len(numerical_data_frame.columns)
    
    def preprocess(i,column,number_of_data):
    
        tmp_dict = {}
        #tmp_df = pd.DataFrame(index = (i,))
        
        if distribution=="normal":
            data                  = np.random.randn(number_of_data)
        elif distribution=="normal":
            data        = np.random.default_rng().chisquare(df,number_of_data)
        else:
            data         = np.random.rand(number_of_data) 
            
        p                     = pearsonr(column,data)
        
        temp_numerical_data_frame  = pd.concat([numerical_data_frame, pd.DataFrame(data, columns=[n_length])], axis = 1)
        
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
    
    tmp_dict=Parallel(n_jobs=-1, verbose=1, pre_dispatch='1.5*n_jobs')(
        delayed(preprocess)(i,column,number_of_data) for i in range(maxiteration))
    

    result = pd.DataFrame()
    new_tmp_dict = {}
    
    for i, d in enumerate(tmp_dict):
        
        index_ =list(d[i].keys())[0]

        p_val           = d[i]['p']
        correlation     = d[i]['correlation']
        cov_sum         = d[i]['cov_sum']
        generated_data  = d[i]['generated_data']
        

        new_tmp_dict[i] ={}
        
        new_tmp_dict[i]['generated_data']  = generated_data
        new_tmp_dict[i]['p_val']           = p_val
        new_tmp_dict[i]['correlation']     = correlation
        new_tmp_dict[i]['cov_sum']         = cov_sum

        tmp_df               = pd.DataFrame(index = (i,))
        tmp_df["i"]          = i
        tmp_df["cov_sum"]    = cov_sum
        tmp_df["p"]          = p_val
        
        result  = pd.concat([result, tmp_df])
        
    result.sort_values(by=["cov_sum"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    found_index =  result.loc[0,"i"]
    
    target_generated_data=new_tmp_dict[found_index]['generated_data']
    found_p=new_tmp_dict[found_index]['p_val']
    correlation=new_tmp_dict[found_index]['correlation']
    
    numerical_data_frame = pd.concat([numerical_data_frame, pd.DataFrame(target_generated_data, columns=[n_length])], axis = 1)
    
    return target_generated_data,correlation,found_p,numerical_data_frame
                        

def create_pearson_numerical_data(column,number_of_data,numerical_data_frame,distribution,correlation,maxiteration= 1000,df =1):
    
    n_length                          = len(numerical_data_frame.columns)
    
    def preprocess(i,column,number_of_data):
    
        tmp_dict = {}
        pearson_correlation = lambda x: abs(correlation - pearsonr(column,x)[0])

        if distribution=="normal":
            data = minimize(pearson_correlation, 
                            np.random.randn(number_of_data)).x    
        elif distribution=="normal":
            data = minimize(pearson_correlation, 
                           np.random.default_rng().chisquare(df,number_of_data)).x  
                
        else:
            data = minimize(pearson_correlation, 
                           np.random.rand(number_of_data)).x   
            
        p                     = pearsonr(column,data)
        
        temp_numerical_data_frame  = pd.concat([numerical_data_frame, pd.DataFrame(data, columns=[n_length])], axis = 1)
        
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
    
    tmp_dict=Parallel(n_jobs=-1, verbose=1, pre_dispatch='1.5*n_jobs')(
        delayed(preprocess)(i,column,number_of_data) for i in range(maxiteration))
    

    result = pd.DataFrame()
    new_tmp_dict = {}
    
    for i, d in enumerate(tmp_dict):
        
        index_ =list(d[i].keys())[0]

        p_val           = d[i]['p']
        correlation     = d[i]['correlation']
        cov_sum         = d[i]['cov_sum']
        generated_data  = d[i]['generated_data']
        

        new_tmp_dict[i] ={}
        
        new_tmp_dict[i]['generated_data']  = generated_data
        new_tmp_dict[i]['p_val']           = p_val
        new_tmp_dict[i]['correlation']     = correlation
        new_tmp_dict[i]['cov_sum']         = cov_sum

        tmp_df               = pd.DataFrame(index = (i,))
        tmp_df["i"]          = i
        tmp_df["cov_sum"]    = cov_sum
        tmp_df["p"]          = p_val
        
        result  = pd.concat([result, tmp_df])
        
    result.sort_values(by=["cov_sum"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    found_index =  result.loc[0,"i"]
    
    target_generated_data=new_tmp_dict[found_index]['generated_data']
    found_p=new_tmp_dict[found_index]['p_val']
    correlation=new_tmp_dict[found_index]['correlation']
    
    numerical_data_frame = pd.concat([numerical_data_frame, pd.DataFrame(target_generated_data, columns=[n_length])], axis = 1)
    
    return target_generated_data,correlation,found_p,numerical_data_frame
                        

def create_data_from_column(
                          numerical_data_frame,
                          data_frame,
                          columnname,
                          correlation=0.7, 
                          corr_type = 'pearsonr',
                          data_type = "numerical",
                          distribution="normal",
                          df = 1,
                          target_p = 0.05,
                          maxiteration = 1000,
                          number_of_categories  = 3,
                          categorical_even      = True
                          ):
    
    #column = data_creator.labels
    
    """create a column that is correlated or not correlated to a given columns
    It has the same number of the given column, it has the given distribution
    """
    #numerical_data_frame

    column  = data_frame[columnname]
    #numerical_data_frame              = pd.DataFrame()
    n_length                          = len(numerical_data_frame.columns)
    
    without_correlation   = 0
    
    start_time = timer()
    number_of_data = len(column)
    
    found_min = ''
    
    # if data_type  == "numerical":
        
        
    if corr_type == "nocorrelation":

        data,correlation,found_p,numerical_data_frame=create_random_numerical_data(column,
                                                                                   number_of_data,
                                                                                   numerical_data_frame,
                                                                                   distribution,
                                                                                   maxiteration= maxiteration,df =df)


    elif corr_type == 'pearsonr':
        
        data,correlation,found_p,numerical_data_frame=create_pearson_numerical_data(column,
                                      number_of_data,
                                      numerical_data_frame,distribution,
                                      correlation,
                                      maxiteration= 10,
                                      df =df)
    
    else:
        print("wrong corrtype")
                 
    # elif data_type == "categorical":
        
    #     data, correlation,found_p,found_min = create_random_categorical_with_p(column,
    #                                     number_of_categories, 
    #                                     target_p = target_p,
    #                                     maxiteration = 10000,
    #                                     categorical_even = categorical_even)
                

          
    elapsed_time = round(timer() - start_time,2) # in seconds
    
    #print(f"corr matrix: {numerical_data_frame.corr()}")
    
    #print(f"create time {elapsed_time} seconds")      
    return data, elapsed_time,correlation,found_p,numerical_data_frame,found_min

