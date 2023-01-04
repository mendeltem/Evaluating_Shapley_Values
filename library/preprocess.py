#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 10:14:44 2022

@author: temuuleu
"""

import pandas as pd
from matplotlib.pyplot import figure

from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import numpy as np
import os

def get_ratio(p1,p2):
    ##get the ratio of 2 numercial values
    
    if p1 >= p2:
        smaller = p2
        bigger  = p1
    else:
        smaller = p1
        bigger  = p2
        
    return smaller / bigger  


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
            

    number_of_data =   shap_values.shape[0]  

    array = np.zeros((number_of_data,len(dictionary)))
    

    for id_x in range(number_of_data):

        for i,k in enumerate(dictionary.keys()):
            if len(dictionary[k]) > 1:
                array[id_x][dictionary[k][k]] = sum(shap_values[id_x][dictionary[k][k+"_old"]])
            else:
                array[id_x][dictionary[k][k]] = shap_values[id_x][dictionary[k][k]]   
    
    return array, list(dictionary.keys())


def get_sample_bin(df,size, start, end, sex ,seed = 0, sex_column = "sex", age_column = "age"):
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
    
    sample_list= (df[sex_column] == sex) & (df[age_column] >= start) & (df[age_column] < end)
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

            
def normalize_data(X):

    # # normalize data (this is important for model convergence)
    dtypes = list(zip(X.dtypes.index, map(str, X.dtypes)))
    for k,dtype in dtypes:
        if dtype == "float64":
            X[k] -= X[k].mean()
            X[k] /= X[k].std()
    
    return X


def get_small_random_namber(max_value = 24):
    
    max_value = abs(max_value)
    
    if max_value:

        min_int = min(np.random.randint(0, max_value), np.random.randint(0, max_value))
        
        return min_int
    else:
        return 0
    
    
    
#creat bins for men and women according to age.
#the number were choosen according to age histogram from men(age) and women(age)

def balance_age_sex(X_DATA,Y_DATA, 
                    searched_p_value = 0.1, 
                    bin_size = 10, 
                    balance_seeds = 1000, 
                    show = True,
                    real_data_path = 'data/real_data/plot/',
                    name = ""
                    ):
    
    Y_DATA = Y_DATA.reset_index(drop=True)
    X_DATA = X_DATA.reset_index(drop=True)
    
    if show:
    
        fig_with = 10
        fig_height = 8

        #plot age sex distribution before it is balanced
        figure(num=None, figsize=(fig_with, fig_height), dpi=80, facecolor='w', edgecolor='k')
        plt.title('Sex')
        plt.hist(X_DATA.loc[X_DATA["sex"] == 0, "sex"], alpha=0.5,facecolor='green' , range=(0,2), bins=2, label = "Male")
        plt.hist(X_DATA.loc[X_DATA["sex"] == 1, "sex"], alpha=0.5,facecolor='red' ,range=(0,2), bins=2, label = "Female")
        plt.legend()
        plt.savefig(os.path.join(real_data_path,f"{name}_age_bar.png"),dpi=100, bbox_inches='tight')
        plt.show()
    
        figure(num=None, figsize=(fig_with, fig_height), dpi=80, facecolor='w', edgecolor='k')
        plt.title('Train Age')
        plt.hist(X_DATA.loc[X_DATA["sex"] == 0, "age"], alpha=0.5,facecolor='green' , bins=15, label = "Male")
        plt.hist(X_DATA.loc[X_DATA["sex"] == 1, "age"], alpha=0.5,facecolor='red' , bins=15, label = "Female")
        plt.legend()
        plt.savefig(os.path.join(real_data_path,f"{name}_age_hist.png"),dpi=100, bbox_inches='tight')
        plt.show()
    
    print(f"datasize before balace: {X_DATA.shape[0]}")

    age_bins = {}

    for age in range(0,110,bin_size):
        
        number_of_men   = sum((X_DATA["sex"] == 0) & (X_DATA["age"] >= age) & (X_DATA["age"] < age+bin_size))
        number_of_women = sum((X_DATA["sex"] == 1) & (X_DATA["age"] >= age) & (X_DATA["age"] < age+bin_size))
        
        #print(f"men   count between {age} and {age+bin_size} : ",number_of_men)
        #print(f"women count between {age} and {age+bin_size} : ",number_of_women)
        
        difference  = number_of_men - number_of_women
        
        age_bins[age] = {}
        age_bins[age]["men"]   = number_of_men
        age_bins[age]["women"] = number_of_women
        
        age_bins[age]["difference"] = difference
        
        
        if difference >0:
            age_bins[age]["smaller_group"] = "women"
        elif difference <0:
            age_bins[age]["smaller_group"] = "men"
        else:
            age_bins[age]["smaller_group"] = ""


    #balancing the data until the difference between the groups reach until p=0.1
    for balance_seed in range(balance_seeds):
        
        balanced_data_df = pd.DataFrame()
        for age_bin, item in age_bins.items():
    
            #print(age_bin)
            number_of_men    = item["men"]  
            number_of_women  = item["women"] 
            difference       = item["difference"] 
            smaller_group     = item["smaller_group"] 
            
            
            temp_difference_threshhold = get_small_random_namber(difference)
        
        
            if smaller_group == "women":
                #print("women")
                women_index = list(get_sample_bin(X_DATA,size = number_of_women, start = age_bin, end = age_bin+bin_size, sex = 1,seed = balance_seed))
                men_index = list(get_sample_bin(X_DATA,size = number_of_women+temp_difference_threshhold, start = age_bin, end = age_bin+bin_size, sex = 0,seed = balance_seed))
                
                balanced_bin_df = pd.concat([X_DATA[women_index] , X_DATA[men_index]])
                balanced_data_df = pd.concat([balanced_data_df, balanced_bin_df])
            elif  smaller_group == "men":  
                #print("men")
                women_index = list(get_sample_bin(X_DATA,size = number_of_men+temp_difference_threshhold, start = age_bin, end = age_bin+bin_size, sex = 1,seed = balance_seed))
                men_index = list(get_sample_bin(X_DATA,size = number_of_men, start = age_bin, end = age_bin+bin_size, sex = 0,seed = balance_seed))
                
                balanced_bin_df = pd.concat([X_DATA[women_index] , X_DATA[men_index]])
                balanced_data_df = pd.concat([balanced_data_df, balanced_bin_df])
                
            else:
                #print("else")
                women_index = list(get_sample_bin(X_DATA,size = number_of_women, start = age_bin, end = age_bin+bin_size, sex = 1,seed = balance_seed))
                men_index = list(get_sample_bin(X_DATA,size = number_of_men, start = age_bin, end = age_bin+bin_size, sex = 0,seed = balance_seed))
                
                balanced_bin_df = pd.concat([X_DATA[women_index] , X_DATA[men_index]])
                balanced_data_df = pd.concat([balanced_data_df, balanced_bin_df])
                
            #print(f"temp_difference_threshhold :  {temp_difference_threshhold}")
                               
        f_balanced = balanced_data_df.loc[balanced_data_df["sex"] == 1, "age"].dropna().to_numpy()
        m_balanced = balanced_data_df.loc[balanced_data_df["sex"] == 0, "age"].dropna().to_numpy()  
        p_value = round(ttest_ind(f_balanced, m_balanced)[1],2)
        
        
        balanced_index = balanced_data_df.index
        balanced_label = Y_DATA[balanced_index]
        
        #print(f"Balanced age    p-Value:  {p_value}")
        if p_value == searched_p_value:break  
        
    print(f"datasize after balace: {balanced_data_df.shape[0]}")
    
    if show:
    
        fig_with = 10
        fig_height = 8
    
        #plot age sex distribution before it is balanced
        figure(num=None, figsize=(fig_with, fig_height), dpi=80, facecolor='w', edgecolor='k')
        plt.title('Balanced Sex')
        plt.hist(balanced_data_df.loc[balanced_data_df["sex"] == 0, "sex"], alpha=0.5,facecolor='green' , range=(0,2), bins=2, label = "Male")
        plt.hist(balanced_data_df.loc[balanced_data_df["sex"] == 1, "sex"], alpha=0.5,facecolor='red' ,range=(0,2), bins=2, label = "Female")
        plt.legend()
        plt.savefig(os.path.join(real_data_path,f"{name}_balanced_age_bar.png"),dpi=100, bbox_inches='tight')
        plt.show()
    
        figure(num=None, figsize=(fig_with, fig_height), dpi=80, facecolor='w', edgecolor='k')
        plt.title('Balanced Train Age')
        plt.hist(balanced_data_df.loc[balanced_data_df["sex"] == 0, "age"], alpha=0.5,facecolor='green' , bins=15, label = "Male")
        plt.hist(balanced_data_df.loc[balanced_data_df["sex"] == 1, "age"], alpha=0.5,facecolor='red' , bins=15, label = "Female")
        plt.legend()
        plt.savefig(os.path.join(real_data_path,f"{name}_balanced_age_hist.png"),dpi=100, bbox_inches='tight')
        plt.show()
    
    
    return balanced_data_df,balanced_label
    