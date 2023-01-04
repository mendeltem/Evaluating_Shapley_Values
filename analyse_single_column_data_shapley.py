#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 16:20:53 2022

@author: temuuleu
"""

import os 
from library.synthetic_creator import SingleDataCreater
import pandas as pd
import configparser
from library.learning_lib import create_dir
import argparse
from collections import Counter
import warnings

import numpy as np
from joblib import Parallel, delayed
from imblearn.over_sampling import SMOTE,RandomOverSampler
from collections import Counter

from sklearn.calibration import CalibratedClassifierCV
import shap
from interpret.blackbox import ShapKernel
import matplotlib.pyplot as plt
from sklearn.preprocessing  import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing  import StandardScaler
from sklearn.linear_model import LinearRegression
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier


from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

from sklearn.metrics import f1_score, accuracy_score

from interpret import show
from interpret import preserve
from datetime import datetime

from sklearn.model_selection import train_test_split
# pip install keras

# pip install tensorflow
from library.learning_lib import create_dir,force_plot_true,shorten_names,collect_hot_encoded_features_kernel
from library.learning_lib import recreate,undummify


from library.preprocess import get_ratio, collect_hot_encoded_features,balance_age_sex

import os


def get_results_shap(shapley_values, sample_size , shap_sample_data_test,
                     y_predicted_sample_data_test,
                     y_test_sample_data_test,
                     all_feature_names
                     ):

    """The function initializes an empty dataframe result_df, which will be used to store the results.
     It then loops through the predictions, generating a dataframe single_explain_df containing the Shapley values
     nd feature values for each prediction.
     The predicted and true target values for each prediction are also included in single_explain_df.

     Arguments:

     shapley_values:               a list or a shap._explanation.Explanation object containing the Shapley values for each prediction.
     sample_size:                  the number of predictions for which Shapley values are being calculated.
     shap_sample_data_test:        a dataframe containing the feature values used to generate the predictions.
     y_predicted_sample_data_test: a series or dataframe containing the predicted target values.
     y_test_sample_data_test:      a series or dataframe containing the true target values.
     all_feature_names:            a list of all feature names.

     """
    
    result_df                               = pd.DataFrame( )  

    ###########
    single_index= 0
    
    for s in range(sample_size):
        
        single_explain_df = pd.DataFrame(index = (single_index,))

        if type(shapley_values) == shap._explanation.Explanation:
            feature_scores     =  shapley_values[s].values
        else:
            feature_scores     =  shapley_values[s]
        feature_values     =  shap_sample_data_test.loc[s,:]

        single_explain_df["predicted"]    = int(y_predicted_sample_data_test.iloc[s,:].values)
        single_explain_df["actual"]       =int(y_test_sample_data_test.iloc[s])
    
        for feature_name,feature_score,feature_value  in zip(all_feature_names, feature_scores,feature_values):
            
            #print(f"{ebm_feature_name:>25}  {round(feature_score,2):>20}")
            single_explain_df["score_"+feature_name] = feature_score
            single_explain_df["value_"+feature_name] = feature_value

        result_df = pd.concat([result_df, single_explain_df])
        single_index+=1
        
    return result_df


def create_single_column_data(added_columns):
    """

    This function generates synthetic data with controlled statistics using an instance of a SingleDataCreater class.
    The function takes a list of tuples added_columns as input and iterates over each tuple,
    calling the add_collumn method of the data_creator object for each one.
    After all of the specified columns have been added,
    the function returns a tuple containing the input columns and the target column of the generated data.
    The input columns are determined by filtering out any columns that contain the string "label" in their name.

    Arguments:

    from_column_name:           the name of the column that the new column will be based on
    new_column_name:            the name of the new column to be added
    distribution:               the distribution of the new column
    correlation:                the correlation of the new column with the from_column_name column
    df:                         the data frame to which the new column should be added

    """
            
    data_creator = SingleDataCreater(name=dataset_name, 
                                     number_of_data =600)
    
    
    for added_column in added_columns:
    
        data_creator.add_collumn(from_column_name= data_creator.label_name,
                                 new_column_name=added_column[0],
                                 distribution=added_column[1], 
                                 correlation= added_column[2],
                                 df= added_column[3]
                                 )
    
    DATA = data_creator.get_data()

    input_columns = [c for c in  list(DATA.columns) if not "label" in c]
    
    return  DATA[input_columns], DATA["label"]
        

variable_types             = ["controll_random","controll_constant","fluid"]
distributions              = ["normal","random","chisquare"]
dfs                        = [0,1,5,10,15]
fluid_correlations         = [0.35, 0.55, 0.75] + [-0.35,-0.55,-0.75]

const_correlations         = [0.7]
with_const_correlation     = [1,0]
dataset_name                = f"synthetic_single"


analyse_name                = "synthetic_single_column_data"

label_index       = 0

step              = 0
seeds             = 100
SEED              = 7
shapley_steps     = 5

stability_index_counter = 0

shap_sample_size        = 30
model_counts            = 3

wb_result_path = "result/synth_data/single_column/"
create_dir(wb_result_path)
all_emb_result_df = pd.DataFrame()
all_result_df = pd.DataFrame()
all_emb_result_df = pd.DataFrame()

how_many_step = seeds * len(dfs)* shapley_steps  * model_counts * len(fluid_correlations) * len(with_const_correlation)

"""This code is using nested for loops to iterate over multiple variables. 
The outermost loop is using the range function to loop over a range of integers, seeds.
 The second loop is using the distribution list if df is 0 its normal distribution if else
 it chisquared distibution. 
 The third loop is using the columns fluid correlation to loop over different sets of correlation. 
 The fourth loop is using the with_const_correlation list to add a variavble with constant correlation of 0.7  """
for seed in range(seeds):
    for df in dfs:
        for correlation in fluid_correlations:
            for w_c in with_const_correlation:
        
                added_columns = []
                added_columns.append( ["controll_random_normal","normal",0,0])
                if w_c == 1:
                    added_columns.append( ["controll_const_normal","normal",0,7])
                        
                        
                if df == 0:
                    added_columns.append( ["controll_fluid_normal","normal",correlation,0])
                    distribution = "normal"
                    
                else:
                    added_columns.append( ["controll_fluid_chisquare","chisquare",correlation,df])
                    
                    distribution = "chisquare"
                    print(added_columns)
                    
                    """here is synthetic data is create given columns from the appended columns"""
                    X_data,Y_data  = create_single_column_data(added_columns)
                    """The Counter class from the collections module is being used to count the occurrences of each 
                    unique value in Y_data. The resulting dictionary has keys representing the unique values in
                     Y_data and values representing the count of each unique value.

                    The get_ratio function is then called with the count of the positive class 
                    (assumed to be labeled 1) and the count of the negative class (assumed to be labeled 0)
                    as arguments. The get_ratio function likely returns the ratio of the positive class to
                    the negative class. The returned value is stored in the porportion variable.
                    """
                    ratio = Counter(Y_data)
                    porportion = get_ratio(ratio[1], ratio[0])

                    """StratifiedKFold is a scikit-learn class for generating stratified k-fold cross-validation 
                    splits of a dataset. It ensures that the proportion of samples for each class is approximately 
                    the same across all folds."""
                    cv_outer   = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

                    """The split method takes two arguments: X_data and Y_data, 
                    which are assumed to be the feature matrix and target vector, respectively."""
                    ci = 0
                    for train_index, test_index in cv_outer.split(X_data, Y_data):
                        
                        all_full_models_df   = pd.DataFrame()                    
                        all_ci_models_df   = pd.DataFrame()
                        """This code is tracking the progress of a loop by calculating the percentage of the total
                          number of steps that have been completed."""
                        percent_complete = round((step / how_many_step)  * 100, 2)
                        
                        os.system("clear")
                        print(f"seed: {seed}")
      
                        #percent_complete = round((step / how_many_step)  * 100, 2)
                        print(f"percent_complete   {percent_complete}    {step} / {how_many_step}" )
 
                        ci+=1
                        """The indices in train_index are used to select the rows of X_data and Y_data that correspond
                         to the training set, and the indices in test_index are used to select the rows 
                         that correspond to the test set. """
                        X_train_full_hot, X_test_full_hot = X_data.iloc[train_index, :], X_data.iloc[test_index, :]
                        y_train, y_test = pd.DataFrame(Y_data).iloc[train_index,0], pd.DataFrame(Y_data).iloc[test_index,0]


                        # Extract the column names of the DataFrame and store them in a list
                        all_feature_names = list(X_train_full_hot.columns)
                        # Store the number of rows in the DataFrame in a variable
                        data_size = len(X_train_full_hot)

                        """The Counter class from the collections module is being used to count the occurrences of each 
                        unique value in Y_data. The resulting dictionary has keys representing the unique values in
                         Y_data and values representing the count of each unique value.

                        The get_ratio function is then called with the count of the positive class 
                        (assumed to be labeled 1) and the count of the negative class (assumed to be labeled 0)
                        as arguments. The get_ratio function likely returns the ratio of the positive class to
                        the negative class. The returned value is stored in the porportion variable.
                        """
                        ratio = Counter(y_train)
                        porportion = get_ratio(ratio[1], ratio[0])
                        number_of_0_label =  Counter(y_train)[0]
                        number_of_1_label =  Counter(y_train)[1]

                        """This code is using the StandardScaler class from the sklearn.preprocessing module 
                           to standardize a dataset. Standardization is a preprocessing step that scales the features of 
                           a dataset so that they have zero mean and unit variance. 
                           This can be useful for some machine learning algorithms that assume that the 
                           features are normally distributed."""
                        scaler = StandardScaler()
                        scaler.fit(X_train_full_hot)
                        X_train_std = scaler.transform(X_train_full_hot)
                        X_test_std = scaler.transform(X_test_full_hot)

                        """The LogisticRegression class takes several hyperparameters as arguments. 
                        The penalty parameter specifies the type of regularization to use in the model. 
                        The l2 value indicates that L2 regularization should be used.
                        The C parameter specifies the inverse of the regularization strength.
                        A smaller value of C corresponds to a stronger regularization."""
                        linear_model  = LogisticRegression(penalty="l2", C=0.1)
                        linear_model = linear_model.fit(X_train_std, y_train)
                        model_name  = "LogisticRegression"
                        """The resulting array, y_pred, contains the predictions made by the logistic regression
                         model on the test set. """
                        y_pred= linear_model.predict(X_test_std)
                        stability_index_counter+=1
                        
                        """The resulting array is indexed using the [:, 1] notation to select the second column,
                        which contains the probability of the positive class.
                        The resulting array, pred_proba_test_auc, contains the probability of the positive class 
                        for each sample in the test set. This array can be used to calculate 
                        the AUC (area under the curve) metric for evaluating the performance of a 
                        binary classification model."""
                        pred_proba_test_auc = linear_model.predict_proba(X_test_std)[:, 1]
                        roc_auc = roc_auc_score(y_test,pred_proba_test_auc)

                        """The resulting array is flattened using the ravel method to create a tuple of four elements: 
                        (tn, fp, fn, tp). The variables tn, fp, fn, and tp represent the number of true negatives, 
                        false positives, false negatives, and true positives, respectively."""
                        tn, fp, fn, tp = confusion_matrix(y_pred, y_test).ravel()

                        """A confusion matrix is a useful tool for evaluating the performance of a binary classification model.
                        The diagonal elements of the matrix represent the number of correctly classified samples,
                        while the off-diagonal elements represent the number of misclassified samples. 
                        The true positive rate (sensitivity or recall) and the true negative rate (specificity)
                        can be calculated using the values in the confusion matrix.
                        The TPR is defined as the number of true positives divided by the number of true positives
                         plus the number of false negatives. It measures the proportion of positive samples that 
                         are correctly classified by the model."""
                        TPR = tp / (tp + fn)
                        #specificity
                        """The TNR is defined as the number of true negatives divided by the number of true negatives
                         plus the number of false positives. It measures the proportion of negative samples that
                          are correctly classified by the model."""
                        TNR = tn / (tn + fp)
                        #balanced accuracy
                        """The BA is defined as the average of the TPR and TNR. It is a balanced measure of 
                        the accuracy of the model, as it takes into account both the ability of 
                        the model to correctly classify positive samples and the ability to correctly 
                        classify negative samples."""
                        BA = round((TPR + TNR) / 2, 2)

                        """The F1 score is a useful metric for evaluating the performance of a binary classification
                         model when the classes are imbalanced, as it takes into account both the precision and
                        the recall of the model. In this code, the F1 score is calculated for
                        the binary classification model and rounded to two decimal places using the round function.
                        "average" parameter to determine how the F1 score is calculated when there are multiple classes.
                        """
                        f1 = round(f1_score(y_test, y_pred, average='macro'),2)
                        print("")
                        print(f"Linear BA {BA}")
                        print(f"Linear roc_auc {round(roc_auc,2) }")
                        print(f"F1 Score {round(f1_score(y_test, y_pred, average='macro'),2)}")
                        print("")
                        """This lambda function can be used, for example, to explain the predictions made by 
                        the logistic regression model using the SHAP library. 
                        The SHAP library uses function approximators to explain the predictions made by machine learning models.
                        The predict_fn lambda function can be passed to the SHAP library as the function approximator 
                        for the logistic regression model."""
                        predict_fn = lambda x: linear_model.predict_proba(x)
                        # %% Create kernel shap explainer
                        """This code is creating a KernelExplainer object from the SHAP (SHapley Additive exPlanations) 
                        library. The KernelExplainer is a class that is used to explain the predictions made
                         by machine learning models using the SHAP values.
                         The KernelExplainer object is used to calculate the SHAP values for the logistic regression model. 
                         The SHAP values are a measure of the importance of each feature in the model's predictions. 
                         They can be used to understand which features are driving the model's predictions and 
                         how each feature contributes to the final prediction.
                         """
                        kernel_explainer = shap.KernelExplainer(predict_fn, data=shap.sample(X_train_full_hot),n_jobs=-1,
                                                                                 feature_names=all_feature_names)
                        
                        """The Explainer object is used to calculate the SHAP values for the logistic regression model 
                        and its generally faster the KernelExplainer.
                        n_jobs: the number of jobs to run in parallel.
                        Setting n_jobs to -1 will use all available cores.
                        """
                        linear_explainer   = shap.Explainer(linear_model, X_train_std, feature_names=all_feature_names, n_jobs=-1)

                        ci_model_result_df = pd.DataFrame()
                        
                        """By looping over a range of values for the sample_seed, this code is sampling multiple 
                        subsets of the test data and the model's predictions, 
                        and then calculating the SHAP values for each subset. 
                        This allows you to get a better estimate of the SHAP values for the model, 
                        as the SHAP values can vary somewhat depending on the specific subset of data that is being used."""
                        for sample_seed in range(shapley_steps):
                            
                            result_df = pd.DataFrame()
                            """This code is sampling a subset of the test data and the model's predictions for that data. 
                            It is doing this by using the sample function from the SHAP library,
                             which randomly selects rows from a dataframe."""
                            shap_sample_data_test      = shap.sample(X_test_full_hot, shap_sample_size, random_state =sample_seed)
                            y_test_sample_data_test      = shap.sample(y_test, shap_sample_size, random_state =sample_seed) 
                            y_predicted_sample_data_test      = shap.sample(pd.DataFrame(y_pred), shap_sample_size, random_state =sample_seed) 
                            
                            shap_sample_data_test         = shap_sample_data_test.reset_index(drop=True)
                            y_predicted_sample_data_test  = y_predicted_sample_data_test.reset_index(drop=True)
                            y_test_sample_data_test       = y_test_sample_data_test.reset_index(drop=True)
                            """The sample_size variable is being used to specify the number of samples that will 
                             be used to calculate the SHAP values for the model."""
                            sample_size = len(shap_sample_data_test)
                            """This code is standardizing the data in the shap_sample_data_test dataframe by applying 
                            the transform method of the StandardScaler object to it. 
                            Standardization is a common preprocessing step that scales 
                            the data so that it has zero mean and unit variance."""
                            shap_sample_data_test_std  = scaler.transform(shap_sample_data_test)
                            """This code block is being used to suppress any warning messages that might be 
                            generated when calculating the SHAP values for the model. 
                            This is often useful when working with large datasets or complex models,
                            as warning messages can sometimes be generated even when the code is working correctly."""
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore")
                                """This code is using the linear_explainer object to calculate the SHAP values 
                                for the logistic regression model.
                                The SHAP values are stored in the all_linear_shap_values variable as an Explanation object, 
                                which is a data structure provided by the SHAP library that contains the calculated 
                                SHAP values and other information about the model's predictions. 
                                The SHAP values can be accessed using the .values attribute of the Explanation object.
                                """
                                all_linear_shap_values = linear_explainer(shap_sample_data_test_std)
                                

                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore")
                                """The kernel_explainer object has a shap_values method that can be used to calculate 
                                the SHAP values for a given dataset. In this case, 
                                the SHAP values are being calculated for the shap_sample_data_test dataframe,
                                 which contains a subset of the test data."""
                                all_kernel_shap_values               =    kernel_explainer.shap_values(shap_sample_data_test,
                                                                         feature_names = all_feature_names)
                                

                            shapley_values                          = all_linear_shap_values
                            """This code is using the get_results_shap function to create a dataframe of SHAP values
                             for a given sample of data. 

                             The function iterates over each sample in the sample data, and for each sample, 
                             it creates a dataframe with the following columns:

                             """
                            temp_df = get_results_shap(shapley_values, sample_size , shap_sample_data_test,
                                                 y_predicted_sample_data_test,
                                                 y_test_sample_data_test,
                                                 all_feature_names)

                            temp_df["explainer"]                  = "linear_explainer_shap"  
                            
                            result_df = pd.concat([result_df,temp_df])

                            """
                            In the case of a binary classification model, 
                            the SHAP values will be a list of two classes, one for each class in the model.
                            The SHAP values for each class represent the contribution of each feature to the model's 
                            prediction for that class. 


                            The SHAP values for the two classes are identical and negative, which
                            it means that the features are contributing equally and negatively to
                            the model's predictions for both classes. 
                            This could indicate that the model is having difficulty making accurate predictions, 
                            or it could be due to other factors such as class imbalance or the nature of the data.
                            """

                            shapley_values                          = all_kernel_shap_values[1]
                            
                            temp_df = get_results_shap(shapley_values, sample_size , shap_sample_data_test,
                                                 y_predicted_sample_data_test,
                                                 y_test_sample_data_test,
                                                 all_feature_names)
                            
                            
                            temp_df["explainer"]                  = "kernel_explainer_shap"  
                            
                            result_df = pd.concat([result_df,temp_df])
                            result_df["sample_step"]                = sample_seed
                            step +=1
                            
                            
                            ci_model_result_df = pd.concat([ci_model_result_df,result_df])

                        """the ci_model_result_df dataframe is being 
                        populated with various evaluation metrics 
                        for a model, including the model name,
                        the type of model (blackbox or not), 
                        the coverage index (ci), balanced accuracy (ba), 
                        F1 score, sensitivity, specificity, true positive (tp), 
                        true negative (tn), false positive (fp), 
                        false negative (fn), and the area under 
                        the receiver operating characteristic curve (roc_auc). """
                        ci_model_result_df["model_name"]                 = model_name
                        ci_model_result_df["blackbox"]                   = 0
                        ci_model_result_df["ci"]                         = ci
                        ci_model_result_df["ba"]                         = round(BA,2) 
                        ci_model_result_df["f1"]                         = round(f1,2) 
                        
                        ci_model_result_df["sensivity"]                  = round(TPR,2)
                        ci_model_result_df["specificity"]                = round(TNR,2)
    
                        ci_model_result_df["tp"]                         = round(TNR,2)
                        ci_model_result_df["tn"]                         = round(tn,2)
                        ci_model_result_df["fp"]                         = round(fp,2)
                        ci_model_result_df["fn"]                         = round(fn,2)
                        ci_model_result_df["specificity"]                = round(TNR,2)
                        ci_model_result_df["roc_auc"]                    = round(roc_auc,2) 
                        
                        all_ci_models_df  = pd.concat([all_ci_models_df,ci_model_result_df])
   
  
                        # %% Tree based model
                        """The RandomForestClassifier is a machine learning model that belongs to the family 
                        of ensemble learning methods. It is a type of classifier that constructs a large number
                         of decision trees and combines their predictions to make a final classification."""
                        rf = RandomForestClassifier()
                        rf.fit(X_train_full_hot, y_train)
                        y_pred = rf.predict(X_test_full_hot)
    
                        model_name  = "RandomForestClassifier"
                        
                        """The predict_proba method of the RandomForestClassifier returns the predicted probability 
                        for each class for each sample in the input array. In this case, you are selecting only
                         the probabilities for the positive class (with index 1) using the indexing [:, 1].
                        """
                        pred_proba_test_auc = rf.predict_proba(X_test_full_hot)[:, 1]
                        roc_auc = roc_auc_score(y_test,pred_proba_test_auc)
                        """The resulting array is flattened using the ravel method to create a tuple of four elements: 
                           (tn, fp, fn, tp). The variables tn, fp, fn, and tp represent the number of true negatives, 
                           false positives, false negatives, and true positives, respectively."""
                        tn, fp, fn, tp = confusion_matrix(y_pred, y_test).ravel()

                        """A confusion matrix is a useful tool for evaluating the performance of a binary classification model.
                        The diagonal elements of the matrix represent the number of correctly classified samples,
                        while the off-diagonal elements represent the number of misclassified samples. 
                        The true positive rate (sensitivity or recall) and the true negative rate (specificity)
                        can be calculated using the values in the confusion matrix."""

                        """The TPR is defined as the number of true positives divided by the number of true positives
                         plus the number of false negatives. It measures the proportion of positive samples that 
                         are correctly classified by the model."""
                        TPR = tp / (tp + fn)
                        # specificity
                        """The TNR is defined as the number of true negatives divided by the number of true negatives
                         plus the number of false positives. It measures the proportion of negative samples that
                          are correctly classified by the model."""
                        TNR = tn / (tn + fp)
                        # balanced accuracy
                        """The BA is defined as the average of the TPR and TNR. It is a balanced measure of 
                        the accuracy of the model, as it takes into account both the ability of 
                        the model to correctly classify positive samples and the ability to correctly 
                        classify negative samples."""
                        BA = round((TPR + TNR) / 2, 2)

                        """The F1 score is a useful metric for evaluating the performance of a binary classification
                         model when the classes are imbalanced, as it takes into account both the precision and
                        the recall of the model. In this code, the F1 score is calculated for
                        the binary classification model and rounded to two decimal places using the round function.
                        "average" parameter to determine how the F1 score is calculated when there are multiple classes.

                        """
                        f1 = round(f1_score(y_test, y_pred, average='macro'),2)
                        print("")
                        print(f"RandomForest {BA}")
                        print(f"roc_auc {round(roc_auc,2) }")
                        print(f"F1 Score {round(f1_score(y_test, y_pred, average='macro'),2)}")
                        print(f"Accuracy {round(accuracy_score(y_test, y_pred),2)}")
                        print("")
                        

                        """The function predict_proba is a method of 
                        the RandomForestClassifier class in scikit-learn,
                        which returns the class probabilities for each sample
                        in x as a 2D numpy array. The number of columns in
                        the array corresponds to the number of classes, 
                        and the rows correspond to the samples."""
                        predict_fn = lambda x: rf.predict_proba(x)
                        """The KernelExplainer class from 
                        the SHAP (SHapley Additive exPlanations) library is 
                        then being used to create a kernel_explainer object. 
                        The KernelExplainer is a model-agnostic method for 
                        explaining the output of any function with
                        a non-linear decision boundary."""
                        kernel_explainer = shap.KernelExplainer(predict_fn, data=shap.sample(X_train_full_hot),n_jobs=-1,
                                                                                  feature_names=all_feature_names)
                        

                        """The TreeExplainer is a model-specific method 
                        for explaining the output of tree-based models 
                        such as random forests, gradient boosted trees, 
                        and decision trees.

                        n_jobs: the number of jobs to run in parallel.
                        Setting n_jobs to -1 will use all available cores.
                        """
                        TreeExplainer = shap.TreeExplainer(rf, feature_names=all_feature_names,n_jobs=-1)
                        ci_model_result_df = pd.DataFrame()
                        
                        """By looping over a range of values for the sample_seed, this code is sampling multiple 
                        subsets of the test data and the model's predictions, 
                        and then calculating the SHAP values for each subset. 
                        This allows you to get a better estimate of the SHAP values for the model, 
                        as the SHAP values can vary somewhat depending on the specific subset of data that is being used."""
                        for sample_seed in range(shapley_steps):
                            """The shap.sample function returns 
                            a random sample of rows from the input dataset. 
                            The sample is selected using the random_state 
                            argument to ensure reproducibility."""
                            print(f"percent_complete   {percent_complete}    {step} / {how_many_step}" )
                            #sample from test
                            shap_sample_data_test      = shap.sample(X_test_full_hot, shap_sample_size, random_state =sample_seed)
                            y_test_sample_data_test      = shap.sample(y_test, shap_sample_size, random_state =sample_seed) 
                            y_predicted_sample_data_test      = shap.sample(pd.DataFrame(y_pred), shap_sample_size, random_state =sample_seed) 
                            
                            shap_sample_data_test         = shap_sample_data_test.reset_index(drop=True)
                            y_predicted_sample_data_test  = y_predicted_sample_data_test.reset_index(drop=True)
                            y_test_sample_data_test       = y_test_sample_data_test.reset_index(drop=True)  
                            
                            sample_size = len(shap_sample_data_test)
                            """
                            The shap_values method returns a list of 
                            SHAP values arrays, one for each sample in
                            the dataset. Each SHAP values array has shape 
                            (num_features,), and represents 
                            the contribution of each feature to 
                            the model's prediction for 
                            the corresponding sample.
                            """
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore")
                                """
                                The kernel_explainer object was created
                                earlier using the shap.KernelExplainer
                                class, which is a model-agnostic method
                                for explaining the output of any function
                                with a non-linear decision boundary.
                                """
                                all_kernel_shap_values               =    kernel_explainer.shap_values(shap_sample_data_test,
                                                                          feature_names = all_feature_names)

                            
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore")
                                """The TreeExplainer object was created 
                                earlier using the shap.TreeExplainer class,
                                which is a model-specific method for
                                explaining the output of tree-based models 
                                such as random forests, gradient boosted
                                trees, and decision trees."""s
                                all_tree_shap_values                    = TreeExplainer.shap_values(shap_sample_data_test)


                            shapley_values                          = all_tree_shap_values[1]
                            """The get_results_shap function is being used
                            to process the SHAP (SHapley Additive exPlanations)
                            values for a single sample and return the results
                            as a DataFrame."""
                            temp_df = get_results_shap(shapley_values, sample_size , shap_sample_data_test,
                                                 y_predicted_sample_data_test,
                                                 y_test_sample_data_test,
                                                 all_feature_names)
   
                            temp_df["explainer"]                  = "TreeExplainer_shap"  
                            
                            result_df = pd.concat([result_df,temp_df])
                            
                            
                            shapley_values                          = all_kernel_shap_values[1]
                            """The get_results_shap function is being used
                            to process the SHAP (SHapley Additive exPlanations)
                            values for a single sample and return the results
                            as a DataFrame.

                            The SHAP values for the two classes are identical and negative, which
                            it means that the features are contributing equally and negatively to
                            the model's predictions for both classes. 
                            This could indicate that the model is having difficulty making accurate predictions, 
                            or it could be due to other factors such as class imbalance or the nature of the data.

                            """
                            temp_df = get_results_shap(shapley_values, sample_size , shap_sample_data_test,
                                                 y_predicted_sample_data_test,
                                                 y_test_sample_data_test,
                                                 all_feature_names)
                            
                            
                            temp_df["explainer"]                  = "kernel_explainer_shap"  
                            
                            result_df = pd.concat([result_df,temp_df])
                            result_df["sample_step"]                = sample_seed
                            step +=1
                            
                            ci_model_result_df = pd.concat([ci_model_result_df,result_df])

                        """the ci_model_result_df dataframe is being 
                        populated with various evaluation metrics 
                        for a model, including the model name,
                        the type of model (blackbox or not), 
                        the coverage index (ci), balanced accuracy (ba), 
                        F1 score, sensitivity, specificity, true positive (tp), 
                        true negative (tn), false positive (fp), 
                        false negative (fn), and the area under 
                        the receiver operating characteristic curve (roc_auc). """
                        ci_model_result_df["model_name"]                 = model_name
                        ci_model_result_df["blackbox"]                   = 0
                        ci_model_result_df["ci"]                         = ci
                        ci_model_result_df["ba"]                         = round(BA,2) 
                        ci_model_result_df["f1"]                         = round(f1,2) 
                        
                        ci_model_result_df["sensivity"]                  = round(TPR,2)
                        ci_model_result_df["specificity"]                = round(TNR,2)
    
                        ci_model_result_df["tp"]                         = round(TNR,2)
                        ci_model_result_df["tn"]                         = round(tn,2)
                        ci_model_result_df["fp"]                         = round(fp,2)
                        ci_model_result_df["fn"]                         = round(fn,2)
                        ci_model_result_df["specificity"]                = round(TNR,2)
                        ci_model_result_df["roc_auc"]                    = round(roc_auc,2) 
                        
                        all_ci_models_df  = pd.concat([all_ci_models_df,ci_model_result_df])
                            
                        """The ExplainableBoostingClassifier is a tree-based 
                        ensemble method for classification tasks. 
                        It combines the predictions of multiple decision trees,
                        each trained on a different subset of the training data,
                        to make a final prediction. 
                        The ExplainableBoostingClassifier is similar
                        to gradient boosting and XGBoost, 
                        but it is designed to be more transparent and 
                        interpretable than other boosting methods.

                        Setting n_jobs to -1 will use all available cores.
                        """
                        ebm = ExplainableBoostingClassifier(random_state=seed, n_jobs=-1)
                        """The fit method trains 
                        the ExplainableBoostingClassifier model on
                        the training data, """
                        ebm.fit(X_train_full_hot, y_train)
                        
                        model_name  = "ExplainableBoostingClassifier"
                
                        y_pred = ebm.predict(X_test_full_hot)
                        """In this code, the predicted class probabilities
                        for the positive class (class 1) are being 
                        selected using the index [:, 1], and the resulting
                        array is being assigned to 
                        the variable pred_proba_test_auc.

                        Then, the roc_auc_score function from scikit-learn 
                        is being used to compute the area under the receiver
                        operating characteristic curve (ROC AUC) 
                        for the predicted class probabilities and 
                        the true labels. """
                        pred_proba_test_auc = ebm.predict_proba(X_test_full_hot)[:, 1]
                        roc_auc = roc_auc_score(y_test,pred_proba_test_auc)
                        """The resulting array is flattened using the ravel 
                        method to create a tuple of four elements: 
                        (tn, fp, fn, tp). The variables tn, fp, fn, and tp
                        represent the number of true negatives, 
                        false positives, false negatives, 
                        and true positives, respectively."""
                        tn, fp, fn, tp = confusion_matrix(y_pred, y_test).ravel()

                        """A confusion matrix is a useful tool for evaluating 
                        the performance of a binary classification model.
                        The diagonal elements of the matrix represent
                        the number of correctly classified samples,
                        while the off-diagonal elements represent 
                        the number of misclassified samples. 
                        The true positive rate (sensitivity or recall) 
                        and the true negative rate (specificity)
                        can be calculated using the values in 
                        the confusion matrix."""

                        """The TPR is defined as the number of true positives 
                        divided by the number of true positives
                         plus the number of false negatives. 
                         It measures the proportion of positive samples that 
                         are correctly classified by the model."""
                        TPR = tp / (tp + fn)
                        # specificity
                        """The TNR is defined as the number of true negatives
                        divided by the number of true negatives
                         plus the number of false positives. It measures 
                         the proportion of negative samples that
                          are correctly classified by the model."""
                        TNR = tn / (tn + fp)
                        # balanced accuracy
                        """The BA is defined as the average of the TPR and TNR.
                        It is a balanced measure of 
                        the accuracy of the model, as it takes into account 
                        both the ability of 
                        the model to correctly classify positive samples and 
                        the ability to correctly 
                        classify negative samples."""
                        BA = round((TPR + TNR) / 2, 2)

                        """The F1 score is a useful metric for evaluating 
                        the performance of a binary classification
                         model when the classes are imbalanced, as it takes 
                         into account both the precision and
                        the recall of the model. In this code, the F1 score 
                        is calculated for
                        the binary classification model and rounded to
                        two decimal places using the round function.
                        "average" parameter to determine how the F1 score is
                        calculated when there are multiple classes.

                        """
                        f1 = round(f1_score(y_test, y_pred, average='macro'),2)
                        
                        print("")
                        print(f"ExplainableBoostingClassifier BA {BA}")
                        print(f"roc_auc {round(roc_auc,2) }")
                        print(f"F1 Score {round(f1_score(y_test, y_pred, average='macro'),2)}")
                        print("")
                        
                        """
                        This code predict_fn which takes an input x and returns the predicted probabilities 
                        for each class using a model called ebm. It then creates an instance of
                        the KernelExplainer class from the shap library, passing in predict_fn as 
                        the first argument and data as the second argument.
                        """
                        predict_fn = lambda x: ebm.predict_proba(x)
                        """
                        The KernelExplainer class is used to approximate SHAP (SHapley Additive exPlanations) values,
                        which are used to explain the output of a machine learning model by identifying the features
                        that contribute most to the model's predictions. The KernelExplainer class uses a technique
                        called kernel SHAP, which uses a weighted linear regression to approximate the SHAP values.
                        """
                        kernel_explainer = shap.KernelExplainer(predict_fn, data=shap.sample(X_train_full_hot),n_jobs=-1,
                                                                                  feature_names=all_feature_names)
                        ci_model_result_df = pd.DataFrame()
                        """By looping over a range of values for the sample_seed, this code is sampling multiple 
                         subsets of the test data and the model's predictions, 
                         and then calculating the SHAP values for each subset. 
                         This allows you to get a better estimate of the SHAP values for the model, 
                         as the SHAP values can vary somewhat depending on the specific subset of data that is being used."""
                        for sample_seed in range(shapley_steps):
                            print(f"percent_complete   {percent_complete}    {step} / {how_many_step}" )
                            
                            shap_sample_data_test      = shap.sample(X_test_full_hot, shap_sample_size, random_state =sample_seed)
                            y_test_sample_data_test      = shap.sample(y_test, shap_sample_size, random_state =sample_seed) 
                            y_predicted_sample_data_test      = shap.sample(pd.DataFrame(y_pred), shap_sample_size, random_state =sample_seed) 
                            
                            shap_sample_data_test         = shap_sample_data_test.reset_index(drop=True)
                            y_predicted_sample_data_test  = y_predicted_sample_data_test.reset_index(drop=True)
                            y_test_sample_data_test       = y_test_sample_data_test.reset_index(drop=True)  
                            
                            """The explain_local method is used to calculate local (instance-level) explanations for
                            a model's predictions. It takes a set of test data as input and returns an explanation for 
                            each instance in the test set. In this case, shap_sample_data_test and 
                            y_test_sample_data_test are likely to be the input data and corresponding 
                            target values for a set of test instances.

                            The exact output of the explain_local method will depend on the specific implementation 
                            of the model. In general, it may return a summary of the model's predictions 
                            for each instance in the test set, along with an explanation of how the model 
                            arrived at those predictions."""
                            ebm_local = ebm.explain_local(shap_sample_data_test,y_test_sample_data_test)
                            ebm_feature_names = ebm_local.feature_names
        
                            size = shap_sample_data_test.shape[0]
        
                            result_df = pd.DataFrame()
                            emb_index= 0
                            """
                             This code is looping through a range of values specified by size and creating a data frame
                             called single_explain_df for each iteration.
                             It then extracts several pieces of information from an object ebm_local, 
                             which is to be a local explanation generated by the explain_local method of a model.
                             """
                            for s in range(size):
                            
                                single_explain_df = pd.DataFrame(index = (emb_index,))
                                
                                feature_scores = ebm_local.data(s)['scores']
                                feature_values = ebm_local.data(s)['values'] 
                                
                                actual = ebm_local.data(s)['perf']['actual']
                                predicted = ebm_local.data(s)['perf']['predicted']
                                
                                single_explain_df["actual"]    = actual
                                single_explain_df["predicted"] = predicted
                                
                                for ebm_feature_name,feature_score,feature_value  in zip(ebm_feature_names, feature_scores,feature_values):
                                    #print(f"{ebm_feature_name:>25}  {round(feature_score,2):>20}")
                                    single_explain_df["score_"+ebm_feature_name] = feature_score
                                    single_explain_df["value_"+ebm_feature_name] = feature_value
                                
                                result_df = pd.concat([result_df, single_explain_df])
                                emb_index+=1
                                
                                
                            result_df["explainer"]                  = "ExplainableBoosting_internal"  
                            ci_model_result_df = pd.concat([ci_model_result_df,result_df])
                            """This code is using the get_results_shap function to create a dataframe of SHAP values
                             for a given sample of data. 

                             The function iterates over each sample in the sample data, and for each sample, 
                             it creates a dataframe with the following columns:

                            The SHAP values for the two classes are identical and negative, which
                            it means that the features are contributing equally and negatively to
                            the model's predictions for both classes. 
                            This could indicate that the model is having difficulty making accurate predictions, 
                            or it could be due to other factors such as class imbalance or the nature of the data.
                             """
                            
                            ##ExplainableBoosting_internal
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore")
                                """
                                The kernel_explainer object was created
                                earlier using the shap.KernelExplainer
                                class, which is a model-agnostic method
                                for explaining the output of any function
                                with a non-linear decision boundary.
                                """
                                all_kernel_shap_values               =    kernel_explainer.shap_values(shap_sample_data_test,
                                                                          feature_names = all_feature_names)

                            """The get_results_shap function is being used
                            to process the SHAP (SHapley Additive exPlanations)
                            values for a single sample and return the results
                            as a DataFrame.

                            The SHAP values for the two classes are identical and negative, which
                            it means that the features are contributing equally and negatively to
                            the model's predictions for both classes. 
                            This could indicate that the model is having difficulty making accurate predictions, 
                            or it could be due to other factors such as class imbalance or the nature of the data.

                            """
                            result_df = get_results_shap(all_kernel_shap_values[1], sample_size , shap_sample_data_test,
                                                 y_predicted_sample_data_test,
                                                 y_test_sample_data_test,
                                                 all_feature_names)
                           
                            result_df["explainer"]                  = "kernel_explainer_shap"  
                            
                            result_df["sample_step"]                = sample_seed
                            step +=1

                            ci_model_result_df = pd.concat([ci_model_result_df,result_df])
                            
                        """the ci_model_result_df dataframe is being 
                        populated with various evaluation metrics 
                        for a model, including the model name,
                        the type of model (blackbox or not), 
                        the coverage index (ci), balanced accuracy (ba), 
                        F1 score, sensitivity, specificity, true positive (tp), 
                        true negative (tn), false positive (fp), 
                        false negative (fn), and the area under 
                        the receiver operating characteristic curve (roc_auc). """
                        ci_model_result_df["model_name"]                 = model_name
                        ci_model_result_df["blackbox"]                   = 0
                        ci_model_result_df["ci"]                         = ci
                        ci_model_result_df["ba"]                         = round(BA,2) 
                        ci_model_result_df["f1"]                         = round(f1,2) 
                        
                        ci_model_result_df["sensivity"]                  = round(TPR,2)
                        ci_model_result_df["specificity"]                = round(TNR,2)
    
                        ci_model_result_df["tp"]                         = round(TNR,2)
                        ci_model_result_df["tn"]                         = round(tn,2)
                        ci_model_result_df["fp"]                         = round(fp,2)
                        ci_model_result_df["fn"]                         = round(fn,2)
                        ci_model_result_df["specificity"]                = round(TNR,2)
                        ci_model_result_df["roc_auc"]                    = round(roc_auc,2) 
                        
                        all_ci_models_df  = pd.concat([all_ci_models_df,ci_model_result_df])
                        

                        all_full_models_df  = pd.concat([all_full_models_df,all_ci_models_df])     
                        all_full_models_df["distribution"]             = distribution
    
                        all_full_models_df["df"]                       = df
                        all_full_models_df["correlation"]              = correlation
                        all_full_models_df["number_of_men"]              = w_c
                        all_full_models_df["shap_sample_size"]           = shap_sample_size
                        all_full_models_df["train_data_size"]            = data_size
                        all_full_models_df["current_time"]               = datetime.now().strftime("%H:%M:%S")
                        all_full_models_df["stability_index_counter"]    = stability_index_counter
                        all_full_models_df["it_seed"]                       = seed

                        """ the to_excel method of a DataFrame 
                             all_emb_result_df is being used to save 
                             the DataFrame to an Excel file."""
                        all_emb_result_df = pd.concat([all_emb_result_df,all_full_models_df])
                        all_emb_result_df.to_excel(wb_result_path+f"part_{analyse_name}_result.xlsx")
                        
                        
