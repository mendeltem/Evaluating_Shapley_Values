#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 22:38:59 2021

@author: temuuleu
"""
import pandas as pd
import os
import configparser
import seaborn as sns
from copy import copy
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import re
import shap
import numpy as np


def calc_shap(shap_values, X, expected_value):
    
    explanation = shap.Explanation(shap_values,
                 base_values=np.array(np.full((len(X), ), np.array(expected_value))),
                 data=X.values,
                 feature_names=list(X.columns))

    return explanation



def get_avg_shap_svm(svm_shap_values, feature_names , expected_value, data = 0):
    """
    create Explanation object from Shapley values
    
    """
    
    avg_shap               =  shap._explanation.Explanation(svm_shap_values[0])

    avg_shap.base_values   =  expected_value
    avg_shap.feature_names =  feature_names
    
    avg_shap.values        = svm_shap_values
    
    if data:
        avg_shap.data          =    data


    return avg_shap



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
    if not os.path.exists(output_path) and is_directory(output_path):
        os.makedirs(output_path)



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
    if not os.path.exists(output_path) and is_directory(output_path):
        os.makedirs(output_path)


def get_minimum_shape(shap_test_list):

    min_shape = 0
    shap_test_list_new= []

    for i,sh in enumerate(shap_test_list):
        for s in sh:
            if i == 0:
                min_shape = s.shape[0]

            if s.shape[0] < min_shape:
                min_shape = s.shape[0]

    for i,sh in enumerate(shap_test_list):
        sub_list = []
        for s in sh:
            sub_list.append(s[:min_shape,:])
        shap_test_list_new.append(sub_list) 
    
    return shap_test_list_new  


def get_minimum_shape(test_abs_values):

    min_shape = 0
    
    for i,sh in enumerate(test_abs_values):
        
        if i == 0:
            min_shape = sh.shape[0]
            
        if sh.shape[0] < min_shape:
            min_shape = sh.shape[0]
            
    return (min_shape, sh.shape[1])


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
    if not os.path.exists(output_path) and is_directory(output_path):
        os.makedirs(output_path)

def p_formula(C,n_permutations):    
    return (C + 1) / (n_permutations + 1)


def get_p_value(perm_values,ba_value):
    #Where C is the number of permutations whose score >= the true score.
    C = sum(perm_values > ba_value)
    n_permutations = len(perm_values)

    return p_formula(C,n_permutations)

