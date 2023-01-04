#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 10:50:30 2020

@author: temuuleu
"""
import os
#import random
import re
import os.path
#import optparse
#import yaml

import numpy as np

import string
import datetime as dt   
import pandas as pd


import matplotlib.pyplot as plt
from skimage.transform import resize
import math


def get_important_colums(input_array , output_array, dataframe, id_col = ["ID", "kskopseu"]):
    """create two dataframes from one dataframe with given arrays, and id list
    
    Arguments:
        input_array : numpy array with given colnames and to be renamed col
        example:
        np.array([
            ["barthel index", "scbig_a"],
            ["mRS", "mRS_a_neu"]])
            
        output_array: numpy array with given colnames and to be renamed col
        dataframe: DataFrame 
        id_col:   ID column to identify 
        
    Return:
        inputdataframe:  DataFrame wiht input columns
        
        outputdataframe: DataFrame wiht output columns
    
    """

    temp_df = dataframe.copy(deep=True)

    rename_array = np.concatenate( ([id_col],input_array, output_array))
    name_dict = {}
    for i in rename_array:
        name_dict[i[1]] = i[0]
    temp_df_renamed = temp_df.rename(columns=name_dict)
    temp_df_renamed = temp_df_renamed.astype({id_col[0]: 'int'})


    #get only the given columns
    new_input_list =[id_col[0]]

    for i in input_array[:,0]:
        if i in temp_df_renamed.columns:
            new_input_list.append(i)

    temp_df_renamed[new_input_list] 

    new_output_list =[id_col[0]]

    for i in output_array[:,0]:
        if i in temp_df_renamed.columns:
            new_output_list.append(i)

    temp_df_renamed[new_output_list]     
    
    return temp_df_renamed[new_input_list], temp_df_renamed[new_output_list]    


def impute_random(dataframe, missing_columns):
    """crete random values in the given dataframe"""
    #random imputation
    for feature in missing_columns:
        dataframe[feature + '_imp'] = dataframe[feature]
        dataframe = random_imputation(dataframe, feature)
    return dataframe

def mu_char(char, n):

    text = ""
    for i in range(n):
        text += char
    return text


def append_string_length(text,lenght = 40, char=" "):
    """the string lenght get appended to have the given length"""

    diff = lenght - len(text)
    for u in range(diff):
        text += char
    return text



def random_imputation(df, feature):

    number_missing = df[feature].isnull().sum()
    observed_values = df.loc[df[feature].notnull(), feature]
    df.loc[df[feature].isnull(), feature + '_imp'] = np.random.choice(observed_values, number_missing, replace = True)
    
    return df


def check_missing_values(dataframe, show = 0,tolerance = 0):
    """Visualize the missing information of the dataframe
    
        
    Arguments:
        
        dataframe: DataFrame 
        show:   int 0: nothing, 1 show      
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
        if dataframe[col].isnull().sum():
            percentage_missing = dataframe[col].isnull().sum() / datalength 
            if percentage_missing >= tolerance:
                missing_col.append(col)
                if show == 1:
                    print("missing     : ", col)
                    print("count       : ", dataframe[col].isnull().sum())
                    print("percentage  : ", percentage_missing)
            else:
                not_missing_col.append(col)
            if show == 2 or show == 4:
                print("missing     : ", col)
                print("count       : ", dataframe[col].isnull().sum())
                print("percentage  : ", percentage_missing)
                
    if show == 1 and missing_col:
        mno.matrix(dataframe[missing_col], figsize = (20,5))
                
                
    if show == 3 or show == 4:            
        mno.matrix(dataframe, figsize = (20,5))

    return missing_col,not_missing_col


def hot_encoding(dataframe, drop_multi_category = 1, showmap  =0 ):
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
    
    
def zscore_normalisation(dataframe, colname):
    """zscore transformation 
    
    Arguments:
        dataframe: DataFrame to change
    Return:
        the changed DataFrame        
    """
    temp_df = dataframe.copy(deep=True)
    temp_df[colname] = (temp_df[colname] - temp_df[colname].mean()) / temp_df[colname].std()
    
    return temp_df

def recreate_dir(path):

    os.system('rm -rf '+path)
    create_dir(path)

def list_minus_list(list1, list2):
    return [c for c in list1 if c not in list2]


def search_for_nii(subject_path):
    
    found_paths_withpatter = []
       
    # for not_pattern in not_label_patter:
    
    for roots,dirs,files in os.walk(subject_path):
        for name in files:
            if name.endswith(".nii") or name.endswith(".nii.gz"):
                found_paths_withpatter.append(path_converter(roots + '/' + name))
                    
    return found_paths_withpatter
    


def search_for_masks(all_nii_files,
                     label_pattern = ["infarct","empty","fluid", "dark","fl_"],
                     not_label_patter = ["cor", "sag","dwi","routine_kopf_T1","T1","chronic","tfl"]):
    """Returns all paths with the given patterns in the directory.
    Also check if there is a found path
    Arguments:
        data_path: directory the search starts from.
        pattern: patter of a file to be recognized.
    Returns:
        a sorted list of all paths starting from the root of the file
        system.
        
        boolean found path
        
        
        chronic masks not in bids
        
        
        if 2 or more masks are found take the one with KV!!!
    """

    found_paths_withpatter = [file for pattern in label_pattern\
                              for file in all_nii_files if pattern.lower() in file.lower() and not len(pattern) == 0]
        
        
    found_paths_wihtout_not = [file for pattern in not_label_patter\
                              for file in all_nii_files if pattern.lower() in file.lower() and not len(pattern) == 0]
        
        
    found_list_ofmasks= list_minus_list(found_paths_withpatter, found_paths_wihtout_not)  
    

    # found_list_ofmasks = ["kvmendel.nii","mendel.nii","super"]
    
    
  #   found_list_ofmasks =['CSB_TRIO_20101019_0811_FLAIR_infarct_MCA.nii.gz',
  # 'CSB_TRIO_20101019_0811_FLAIR_infarct_MCA_KV.nii.gz',
  # 'CSB_TRIO_20101019_0811_t2_tirm_tra_dark-fluid_20101019000805_6.nii.gz',
  # 'CSB_TRIO_20111215_0809_t2_tirm_tra_dark-fluid_20111215000805_7.nii.gz',
  # 'CSB_TRIO_20101019_0811_t2_tirm_tra_dark-fluid_20101019000805_6.nii.gz',
  # 'CSB_TRIO_20111215_0809_t2_tirm_tra_dark-fluid_20111215000805_7.nii.gz']


    kv_index = search_in_list("kv",found_list_ofmasks)
    

    if(len(found_list_ofmasks) >= 2) and len(kv_index) == 1:
        
        ur = [found_list_ofmasks[kv_index[0]].replace("_KV","")]

        return [[u for u in found_list_ofmasks if not ur[0] == u]][0]
    
    elif len(kv_index) > 1:
        return []
    else:
        return found_list_ofmasks


def search_in_list(pattern = "kv", liste = [""]):

    return [i for i,k  in enumerate(liste) if pattern in k.lower()]
    



    
    
    # ur = [liste[o].replace(pattern,"") for o in k_index]
         
    
    
    # return [u for u in liste if not ur[0] in u]


        

    
def search_for_dir(all_paths,
                     label_pattern = ["flair","dark","fl_"],
                     not_label_patter = ["cor", "sag","tfl", "lobi"]):
    """Returns all paths with the given patterns in the directory.
    Also check if there is a found path
    Arguments:
        data_path: directory the search starts from.
        pattern: patter of a file to be recognized.
    Returns:
        a sorted list of all paths starting from the root of the file
        system.
        
        boolean found path
        
        
        chronic masks not in bids
        if 2 or more masks are found take the one with KV
    """
    
    # all_paths = ['H.Z. Seq.7 t2_tirm_tra_dark-fluid',
    #          '0015_FL_tra_2.5mm_LoBi_pre']
    
    all_dirs = [path for path in all_paths if is_directory(path)]
  
    found_paths_withpatter = [file for pattern in label_pattern\
                              for file in all_dirs if pattern.lower() \
                                  in file.lower() and not len(pattern) == 0 \
                                      and is_directory(pattern)]
        
        
    found_paths_wihtout_not = [file for pattern in not_label_patter\
                              for file in all_dirs if pattern.lower() \
                                  in file.lower() and not len(pattern) == 0\
                                      and is_directory(pattern)]
        
        
    return list(set(list_minus_list(found_paths_withpatter, found_paths_wihtout_not)))


def find_directory_flair(session_path, label =["flair","dark",]):
        
    #subject_path+"/"+session_name
    found_class_path=""
    found_class_directory_name=""
    
    class_paths = []
    
    class_names = []
    
    for class_index, class_directory in enumerate(os.listdir(session_path)): 
    
        if is_directory(class_directory): 
            
            if "flair" in class_directory.lower() \
                and not "cor" in class_directory.lower() \
                    and not "sag" in class_directory.lower():
                
                class_path = os.path.join(session_path, class_directory)
                if len(get_feature_paths(class_path)) > 0: 
                    found_class_path = class_path
                    found_class_directory_name = class_directory
                    
                    class_paths.append(class_path)
                    
                    class_names.append(os.path.basename(class_path))
                    
            elif "dark" in class_directory.lower()\
                and not "cor" in class_directory.lower() and not "sag" in class_directory.lower():
                class_path = os.path.join(session_path, class_directory)
                
                if len(get_feature_paths(class_path)) > 0: 
                    found_class_path = class_path
                    found_class_directory_name = class_directory
                    
                    class_paths.append(class_path)
                    class_names.append(os.path.basename(class_path))
    
            elif "fl_" in class_directory.lower()\
                and not "cor" in class_directory.lower() and not "sag" in class_directory.lower():
                class_path = os.path.join(session_path, class_directory)
                
                if len(get_feature_paths(class_path)) > 0: 
                    found_class_path = class_path
                    found_class_directory_name = class_directory
                    
                    class_paths.append(class_path)
                    class_names.append(os.path.basename(class_path))
                    
                        #print(class_index, " : ", class_directory)
            # else:
            #     found_class_path=""
            #     found_class_directory_name=""
                 
    return class_paths,found_class_directory_name, class_names


def get_column_names_from_dcm(dataset):            
                
    columns = []           
    
    dont_print = ['Pixel Data', 'File Meta Information Version']
                    
    for data_element in dataset:
        #print(data_element.name)               
        if data_element.VR == "SQ":   # a sequence
            #print(indent_string, data_element.name)
            pass

        else:
            if data_element.name in dont_print:
                pass
                
                #print("""<item not printed -- in the "don't print" list>""")
            else:
                repr_value = repr(data_element.value)
                if len(repr_value) > 50:
                    repr_value = repr_value[:50] + "..."
                if "Unknown" not in  data_element.name:    
                    print(data_element.name)  
                    
                    columns.append(data_element.name)
                    
    return columns
                
  
def isNaN(num):
    return num != num
              
                
def replace_typeofmri_name(text, key ,replacer):

    if key in text.lower():
        return text.replace(re.search("[a-zA-Z_0-9]*", text)[0], replacer)
    else:
        return text
                
def date_string_slipp(file_directory):
    """convert datename with text to a name without text """
    
    pattern =  "[0-9]*"
                    
    date = re.findall(pattern,file_directory)

    new_date = ""
    first = 0;
    
    for index_date, d in enumerate(date):
        if d and first==0:
            new_date=str(d)
            first = 1;
        elif d:
            new_date+=str(d)
            
    return new_date


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


def path_converter(text_path):
    """Convert a path with a space into space with backslash"""

    out = ''
    out_list = []

    for i, t in enumerate(text_path):
        
        if t == ' ':
            out_list.append('\\ ')
        elif t == "(" :
            out_list.append('\(')
        elif t == ")" :
            out_list.append('\)')   
        else:
            out_list.append(t)
        
    return out.join(out_list)


def remove_first_digits(text_path):
    """Convert a path with a space into space with backslash"""

    out = ''

    for i, t in enumerate(text_path):
        
        if t == '_':
            out = text_path[i+1:]
            break
 
    return out


def get_feature_paths(start_dir, extensions = ['dcm']):
    """Returns all image paths with the given extensions in the directory.
    Arguments:
        start_dir: directory the search starts from.
        extensions: extensions of image file to be recognized.
    Returns:
        a sorted list of all image paths starting from the root of the file
        system.
    """
    if start_dir is None:
        start_dir = os.getcwd()
    img_paths = []
    for roots,dirs,files in os.walk(start_dir):
        for name in files:
            for e in extensions:
                if name.endswith('.' + e):
                    img_paths.append(roots + '/' + name)
    img_paths.sort()
    return img_paths


def get_nummeric_only(text, to_int= 0):
    """Get only nummer from the string """

    nummeric_string =""
    
    for character in text:
        if character.isnumeric():
           
            nummeric_string+=character
    if to_int:  
        if len(nummeric_string) == 0:
            return 0
        else:
            return int(nummeric_string)
    else:
        return nummeric_string 

                                
def delete_first_zeros(digit_with_zeros):     
    """Deleting the first zeros from string """       
                
    digit_without_zeros = ""

    snap = 1
    
    d = 0

    for d in digit_with_zeros:

        if d != "0":
            snap = 0
        if snap == 0:
            digit_without_zeros +=d
            
    return digit_without_zeros
                        

def convert_date(date):
    """converting given date structur to datetime
    Arguments:
        date: a string with date and time in it
        
        date = '201004131854'
        
    Returns:
        datetime: datetime  
        
        datetime.datetime(2010, 4, 13, 18, 54)

    """
    date = get_nummeric_only(date)    
    
    
    if len(date) == 8:

        year     =  int(date[:4])  
        month    =  int(date[4:6])
        day      =  int(date[6:8])
                
        date_time = dt.datetime(year,month,day)
        
        return date_time
        
    if len(date) == 12 or len(date) == 14:

        year     =  int(date[:4])  
        month    =  int(date[4:6])
        day      =  int(date[6:8])
        hour     =  int(date[8:10])
        minute   =  int(date[10:12])
        
        date_time = dt.datetime(year,month,day, hour, minute)
        
        return date_time
    else:
        return 0
                 
    
def search_for_niftii(path):
                
    files = []
    
    for f_inexed , file in enumerate(os.listdir(path)):
        
        if re.search(".nii$", file) or re.search(".nii.gz$", file):
            files.append(file)
    
    return files


def check_session_dir(subject_path, correct_mask):
    
    #correct_mask = search_for_masks(subject_path)
    
    correct_mask = search_for_masks(search_for_nii(subject_path))
    
    
    """Check if the correct directory is given"""

    list_session_dir = []
    
    list_session_dir_with_mask = []
    
    found_mask_paths = []
    
    mask_names = []
    
    if not len(get_feature_paths(subject_path)) > 0:
        return "no files found","",[],[]
    
    if not is_directory(subject_path):
        return "not a directory","", [],[]
    
    #get all directory
    list_session_dir = [file_directory for file_index, file_directory\
       in enumerate(os.listdir(subject_path)) if is_directory(file_directory) and "csb" in file_directory.lower()]
    
    try:
        
        #get the session that hast at least the same name with mask session
        for dir_name in list_session_dir:
            for mask_path in correct_mask:
               if get_nummeric_only(mask_path.lower()) in get_nummeric_only(mask_path.lower()) \
                   or get_nummeric_only(mask_path.lower()) in get_nummeric_only(dir_name.lower()):    
                   list_session_dir_with_mask.append(dir_name) 
                   
        list_session_dir_with_mask.sort()   
        
        
        if list_session_dir_with_mask:
            for mask_path in correct_mask:
                if get_nummeric_only(list_session_dir_with_mask[0]) in get_nummeric_only(os.path.basename(mask_path))\
                   or get_nummeric_only(os.path.basename(mask_path)) in get_nummeric_only(list_session_dir_with_mask[0]):

                    mask_names.append(os.path.basename(mask_path))
                    found_mask_paths.append(mask_path)
    
        if list_session_dir_with_mask and found_mask_paths:
            return "",list_session_dir_with_mask[0],list(set(mask_names)), list(set(found_mask_paths))
        else:
            return "no masks found","",[],[]
    
    except:
        return "Failed","",[],[]

    
def elimenate_quote(word):
    """eliminating quote  like " and ' """

    for i, c in enumerate(word):
        if i==0:
            begin = c
        end = c 
        
    if begin == '"' and end == '"':
        return word[1:-1]
    if begin == "'" and end == "'":
        return word[1:-1] 
        
    else:
        return word 
    
# info2 = get_sessions(df_manual, "Ordner ID", int(subject_directory))

def get_sessions(df, key ,subject_id):
    
    
    """"""
    b = False
    c = False
    
    o = 0
    
    # df = df_manual
    # key="Ordner ID"
    # subject_id = 1000005
    
    for i, value in enumerate(df[key]):
        if value == subject_id:
            b  = True
            index = df.loc[df[key] == value].index.item()
           
        if isNaN(value) and b:
            c = True
            o +=1
            
        if not isNaN(value) and b and c:
            break
        
    if b:
        return df.iloc[index:index+o+1]  
    else:
        return pd.DataFrame()
     

def myprint(dataset, indent=0):
    """Go through all items in the dataset and print them with custom format

    Modelled after Dataset._pretty_str()
    """
    dont_print = ['Pixel Data', 'File Meta Information Version']

    indent_string = "   " * indent
    next_indent_string = "   " * (indent + 1)

    for data_element in dataset:
        if data_element.VR == "SQ":   # a sequence
            print(indent_string, data_element.name)
            for sequence_item in data_element.value:
                myprint(sequence_item, indent + 1)
                print(next_indent_string + "---------")
        else:
            if data_element.name in dont_print:
                print("""<item not printed -- in the "don't print" list>""")
            else:
                repr_value = repr(data_element.value)
                if len(repr_value) > 50:
                    repr_value = repr_value[:50] + "..."
                print("{0:s} {1:s} = {2:s}".format(indent_string,
                                                   data_element.name,
                                                   repr_value))


def reshape_3d_image(images,
                     channels,
                     width,height, 
                     mode = "reflect",
                     preserve_range=True,
                     anti_aliasing=True):
    """reshape slices of images and reverse the
    channels to the first index place
    
    """
    new_images = np.zeros((channels,width, height))

    for index in range(images.shape[2]):
        
        single_image = images[:,:,index]
        
        
        #plt.imshow(single_image)
        
        
        resized_image = resize(single_image,(width, height),
                           mode = mode,
                           preserve_range = preserve_range,
                           anti_aliasing=anti_aliasing)
        
        new_images[index] = resized_image
        
    return new_images


def sort_directory(path):
    """sort directorys nummerical 
    
    """
    
    directorys = os.listdir(path)
    
    direct_dict = {}

    new_directory = []
    
    for i,directory in enumerate(directorys):
    
        if is_directory(directory):
        
            number = get_nummeric_only(directory, int)
    
            direct_dict[number] = i
        else:
            number = get_nummeric_only(directory, str)
    
            direct_dict[number] = i
       
    for key, item in direct_dict.items():
        new_directory.append(directorys[item])
            
    return new_directory



def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
        

def skullstripping(input_img, output_path, filename):
    
    """
    Extracts brain from surrounding skull.
    
    INPUTS:
    input_img: (an xisting .nii.gz file) input image
    output_path: (an existing path) path to save output image
    filename: (a file name) output file name
    
    OUTPUTS:
    filename: skullstripped input_img saved to output_path
    """
    
    from nipype.interfaces import fsl

    btr = fsl.BET()
    btr.inputs.in_file = input_img
    btr.inputs.frac = 0.4
    btr.inputs.out_file = output_path +"/" +filename + "_stripped" + ".nii.gz"
    res = btr.run() 
    

def normalize(nifti_img, mask, template, output_image_path, output_mask_path,filename,mask_index):
    
    """
    Registers input image and lesion mask to template.
    
    INPUTS:
    nifti_img: (existing .nii.gz file) input image
    mask: (existing .nii.gz mask of nifti_img) mask file 
    template: (an existing .nii.gz template file) template to register nifti_img and mask to
    output_path: (an existing path) path to save output image and mask to
    filename: (a file name) output file name
    
    OUTPUT:
    filename: normalized nifti_img and mask saved to output_path
    """
    
    import ants
    
    fi = ants.image_read(template)
    mi = ants.image_read(nifti_img)
    img_mask = ants.image_read(mask)

    mytx = ants.registration(fixed=fi, 
                             moving=mi, 
                             outprefix = "normalized_",
                             type_of_transform = 'Affine')
    
    mywarpedimage = ants.apply_transforms(fixed=fi, 
                                      moving=mi,
                                      transformlist=mytx['fwdtransforms'])
    
    mywarpedmask = ants.apply_transforms(fixed=fi, 
                                      moving=img_mask,
                                      transformlist=mytx['fwdtransforms'])
    
    img_data = mywarpedimage.numpy()
    mask_data = mywarpedmask.numpy()
    
    nii_img = nib.nifti1.Nifti1Image(img_data, affine=None, header=None, extra=None, file_map=None)
    nii_mask = nib.nifti1.Nifti1Image(mask_data, affine=None, header=None, extra=None, file_map=None)
    
       
    nib.save(nii_img, output_image_path +"/" +filename + "_image.nii.gz")
    nib.save(nii_mask, output_mask_path +"/" +filename + "_mask"+str(mask_index)+ ".nii.gz")
    
    return nii_img, nii_mask

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
