#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 19:28:02 2022

@author: temuuleu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os

from itertools import combinations
from scipy.stats import ttest_ind
from copy import copy
import seaborn as sns
from os.path import join

""" in_dict is a dictionary that maps keys to values, where the keys are the names of different variables
Keys are normal names that are used in the medicin and the values are short names
"""
in_dict = {
 'Age': 'age',#
 'Alcohol': 's_hxalk_a',#
 'Hypercholesterolemia': 'hxchol_ps',#
 'Glucose': 'pmglu',#
 'Cholesterol': 'pmchol',#
 'hsCRP': 'hsCRP',  #
 'Sex': 'sex',#
 "Smoking":"hxsmok_a",#
 'Education [years]': 'i_bi_y', #
 'TOAST':'s_kltoas_TL2',#
 "Barthel Index":"scbig_a"#
}

new_in_dict = {}
for key in in_dict.keys():
    new_in_dict[in_dict[key]] = key

def create_dir(output_path):
    """creates a directory of the given path"""
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        

def ttest_groups_scores(df, group_column_name,score_columns):
    """
    This is a function that performs a t-test on two groups of scores, and returns the results as a data frame.

    df:                     a data frame containing the data.
    group_column_name:      a string indicating the name of the column in df that contains the group names.
    score_columns:          a list of strings indicating the names of the columns in df that contain the scores.
    """

    score_column = "scores"

    score_mean = df[score_columns].mean(axis=1).reset_index(drop=True)
    score_mean.name=score_column
    explaienrs = df[group_column_name].reset_index(drop=True)
    
    mean_axis_1_scores_df = pd.concat([score_mean,explaienrs], axis=1)
    test_df = mean_axis_1_scores_df.dropna()

    """
    the function computes the mean score for each row (across the columns specified in score_columns)
    and stores it in a new column called 'scores'. It also stores the group names in a separate column called explaienrs.
     Then, it removes any rows with missing values from the resulting data frame.
    
    The function groups the data by the group names and calculates the mean and standard deviation of
    the scores for each group. It renames the columns containing these statistics to 'mean_std' and 'std_std',
    respectively.
    """
    grouped_mean_std = test_df.groupby(group_column_name).mean()
    grouped_std_std = test_df.groupby(group_column_name).std()
    grouped_mean_std = grouped_mean_std.rename(columns={"scores" : "mean_std"})
    grouped_std_std = grouped_std_std.rename(columns={"scores" : "std_std"})

    concat_group = pd.concat([grouped_mean_std,grouped_std_std], axis=1).round(2)
    
    grps = test_df[group_column_name].unique()
    combs = combinations(grps, 2)

    """    The function then performs a t-test on all pairs of groups, using the ttest_ind function from
    the scipy.stats module. It stores the p-values of these tests in a dictionary, with keys of the form 'c1_c2',
     where c1 and c2 are the names of the two groups being compared."""
    
    ttests = {
        f'{c1}_{c2}': ttest_ind(
            test_df.loc[test_df[group_column_name] == c1, score_column], 
            test_df.loc[test_df[group_column_name] == c2, score_column]
        ) for c1, c2 in combs
    }
    
    df_result = pd.DataFrame(index=(score_column,))

    """the function converts the dictionary of p-values into a data frame, 
    with one row and one column for each pair of groups. It returns this data frame, 
    as well as a data frame containing the mean and standard deviation statistics for each group."""
    
    for ti,(k,item) in enumerate(ttests.items()):
        df_result[k+"_pvalue"] =np.round( item[1],2)
        
    return df_result,concat_group
         

def normalize_2d(matrix):
    """This is a function that normalizes a 2D matrix using the L1 norm."""
    # Only this is changed to use 2-norm put 2 instead of 1
    norm = np.linalg.norm(matrix, 1)
    # normalized matrix
    matrix = matrix/norm 
    return matrix


def normalize_columns(df,score_columns):
    """
    This is a function that normalizes the columns of a data frame using the L1 norm.

    df:              a data frame containing the data to be normalized.
    score_columns:   a list of strings indicating the names of the columns in df that should be normalized.

    """


    """
    The function first makes a copy of the input data frame and resets its index. 
    It then separates the data frame into two parts: one containing the columns to be normalized (score_df), 
    and the other containing the rest of the columns (rest_df).
    """
    copy_df = copy(df)
    copy_df = copy_df.reset_index(drop=True)

    all_col = list(copy_df.columns)
    rest_columns = [f for f in all_col if not f in score_columns]

    rest_df =         copy_df[rest_columns]
    score_df = copy_df[score_columns]
    score_columns = list(score_df.columns)
    
    r,c =score_df.shape
    normal_df  = pd.DataFrame()

    """the function iterates over the rows of score_df, normalizes each row using the normalize_2d 
    function (which normalizes a matrix using the L1 norm), and stores the normalized values back in the data frame.
    
    The function concatenates score_df and rest_df back into a single data frame, and returns the result
    """
    for i in range(r):
        row_values = score_df.iloc[i,:].values
        normalized_row = np.round(normalize_2d(row_values),2)
        
        for z, score_column in enumerate(score_columns):
            score_df.loc[i,score_column] = np.round(normalized_row[z],2)
            
    return pd.concat([score_df, rest_df],axis =1)


def create_plot_df(data_df):
    """
     The function processes the data frame and creates a new data frame containing three columns: 'Feature_name',
    'Value', and 'std'. It also creates a dictionary that maps 'Feature_name' values to colors.

     The function first initializes an empty list data and an empty dictionary colors.
     It then iterates over the columns of the input data frame, extracting the string and number from each column tuple.
     If the string is not in the list of score columns, the function skips this column.

     For each column tuple, the function adds a dictionary entry to data_dict,
     with the string as the key and the number as the value.
     It also assigns a color to the string, stored in the colors dictionary.

     After processing all the columns, the function creates the new data frame by iterating over
     the entries in data_dict and appending the relevant information to data. Finally,
     it returns the new data frame and the colors dictionary.

    """

    data = []
    colors = {}

    list_of_color = ["red","blue","green","yellow", "brown","red","blue","green","yellow", "brown","yellow","blue","green","yellow", "brown"]
    data_dict = {}
    
    for cols,k in data_df.items():
        data_sub = []
        Feature_name = cols[0]
        
        if not Feature_name in score_columns:continue
        data_dict.setdefault(Feature_name,{})
        value_name = cols[1]
        value = k.values[0]
        data_dict[Feature_name][value_name]  = value

    color_index = 0
    for k, item in data_dict.items():
        
        temp = []
        colors[k] = list_of_color[color_index]
        
        temp.append(k)
        temp.append(item['mean'])
        temp.append(item['std'])
        data+=[temp]
        color_index +=1
    
    plot_df = pd.DataFrame( data,columns = ['Feature_name','Value', 'std']) 
     
    return plot_df,colors
                
        
ml_steps         = ['it_seed', 'ci','sample_step']
model_explaienr  =   ['model_name','explainer_name']

real_result_path = "real"

real_data_statistical_path                 = os.path.join(real_result_path,"statistical/")
create_dir(real_data_statistical_path)

results  = [join(real_result_path,f) for f in os.listdir(real_result_path) if "." in f and not "opie" in f]


all_result_df = pd.DataFrame()


scores_cols = ['Sex', 'Alcohol', 'Hypercholesterolemia', 'Cholesterol', 'Glucose',
       'Age', 'Education [years]']

for r_i,result_path in enumerate(results):

    # Extract the base name of the file from the file path and store it in a variable
    result_name = os.path.basename(result_path).split(".")[0]
    # Extract the first character of the file name and store it in a variable
    type_columns    = result_name[:1]
    # Extract the 13th and 14th characters of the file name and store them in a variable
    gender_balance  = result_name[13:14]
    # Extract the last character of the file name and store it in a variable
    balance_balance = result_name[-1:]
    # Read the file into a DataFrame, setting the first column as the index
    g_balanced_1_l_balanced_1_df                             = pd.read_excel(result_path, index_col=0)
    # Extract a list of column names from the DataFrame that contain the word "score" and do not contain the "&" character
    old_score_columns = [sc for sc in  g_balanced_1_l_balanced_1_df.columns if "score" in sc and  not "&" in sc]  

    rn_dict = {}
    # Iterate over the list of column names and add key-value pairs to the dictionary where the keys are the column
    # names and the values are the column names with the first six characters removed
    for sc in old_score_columns:
        nn =sc[6:]
        if nn in new_in_dict.keys():
            rn_dict[sc] = new_in_dict[nn]

    #rename the socre columns
    score_columns    = [rn_dict[sc] for sc in  old_score_columns  if sc in rn_dict.keys() ]
    g_balanced_1_l_balanced_1_df  = g_balanced_1_l_balanced_1_df.rename(columns=rn_dict)
    # Normalize the columns of the DataFrame using the function normalize_columns
    normal_df  = normalize_columns(g_balanced_1_l_balanced_1_df,score_columns)
    # Perform a t-test on the columns of the DataFrame using the function ttest_groups_scores
    A_ttest_p_value,grouped_mean_std           =     ttest_groups_scores(normal_df, "explainer_name", score_columns)


    """Grouping the data in the DataFrame normal_df by the "explainer_name" column and aggregating the data by
     mean and standard deviation for the columns in the list score_columns. 
     The resulting DataFrame is stored in A_score_difference_df.
     
     Get the score values
     """
    A_score_difference_df          =    normal_df.groupby("explainer_name")[score_columns].agg(["mean","std"]).round(2)
    A_score_difference_mean_df     =    normal_df.groupby("explainer_name")[score_columns].mean().round(2).T.reset_index()

    #getting the top score columns
    top_explanation_scores         =    list(A_score_difference_mean_df.sort_values("ebm_explanation")["index"][-5:].values)[::-1]
    A_score_difference_df          =    normal_df.groupby("explainer_name")[top_explanation_scores].agg(["mean","std"]).round(2)

    #get the balanced accuracy values
    mean_ba  = np.round(normal_df["ba"].mean(),2)
    mean_std = np.round(normal_df["ba"].std(),2)

    part_df = pd.DataFrame(index = (r_i,))
     
    type_columns    = result_name[:1]
    gender_balance  = result_name[13:14]
    balance_balance = result_name[-1:]
    
    A_score_difference_df["p value"] =     A_ttest_p_value.values[0][0]
    A_score_difference_df["gender balance"] = int(gender_balance)
    A_score_difference_df["label balance"] = int(balance_balance)
    A_score_difference_df["number of columns"] = int(type_columns)
    A_score_difference_df["ba_mean"] = mean_ba
    A_score_difference_df["ba_std"] = mean_std

    A_score_difference_df = pd.concat([A_score_difference_df,grouped_mean_std], axis=1)

    #rename the medical short name with real names
    new_df = pd.DataFrame()

    for col in A_score_difference_df.columns:
        if not type(col) == str:
            print(type(col))
            colname = col[0] +"_"+ col[1]
        else:
            colname = col
        
        columns_df = A_score_difference_df[col]
        columns_df.name = colname
        print(colname)
        new_df = pd.concat([new_df,columns_df],axis=1)

        type_columns    = result_name[:1]
        gender_balance  = result_name[13:14]
        balance_balance = result_name[-1:]
        
    new_df.to_excel(real_data_statistical_path+f"t{type_columns}_g{gender_balance}_b{balance_balance}_mean_diff.xlsx")
    
