import ast
import pandas as pd
import numpy as np
import os

def remove_rows_with_missing_ratings(input_dataset): #remove property data with missing values in given columns
    for category in ['Accuracy_rating','Location_rating','Check-in_rating','Value_rating']:
        input_dataset = input_dataset[ ~( input_dataset[ category ].isna() )]
    return input_dataset

def combine_description_string(input_dataset): #extract description string and clean
    input_dataset = input_dataset[ ~(input_dataset['Description'].isna()) ] #define dataframe as all entries has a non empty 'Description' section
    
    shifted_property = input_dataset[ ~( input_dataset['Description'].str.contains('About this space') ) ] #every row that doesnt have 'About this space' in its decription
    for category in range(3,19): #for columns which have been shifted
        shifted_property.iloc[0,category] = shifted_property.iloc[0,category+1] #redefine as vlaue of next cell
    input_dataset[input_dataset['ID'] == shifted_property['ID'].item()] = shifted_property #redefine row of original database with corrected row
    input_dataset = input_dataset.iloc[:, :-1] #remove last column which is created due to the error shift in data for that one property

    description_list = input_dataset['Description'].tolist() # convert column to python list
    cleanned_description_list = []
    for description in description_list:
        description = ast.literal_eval(description)
        description.remove('About this space')
        description = [entries for entries in description if entries != '']
        description = " ".join(description)
        cleanned_description_list.append(description) #appened cleanned description into list
    input_dataset.Description = cleanned_description_list # redefine whole whole with the list values
    return input_dataset

def set_default_feature_values(input_dataset): #insert missing values for the given columns with 1
    for column_name in ['guests','beds','bathrooms','bedrooms']: #column names to check for missing values
        input_dataset.loc[input_dataset[column_name].isna(), column_name] = 1 #when no value is present, set value to 1
    return input_dataset

def clean_tabular_data(input_dataset): #perform the functions which clean the tabular data
    removed_missing_ratings = remove_rows_with_missing_ratings(input_dataset)
    combined_descriptions = combine_description_string(removed_missing_ratings)
    cleaned_data = set_default_feature_values(combined_descriptions)
    return cleaned_data

def load_airbnb(input_dataset,column_label): #create tuple of column header and it's values
    column_features = input_dataset[column_label][input_dataset[column_label] != column_label]
    return (column_features,column_label)

if __name__ == '__main__': #run code is script
    os.chdir('C:/Users/Daniel H/Desktop/AI Core/Python/Airbnb/')
    original_dataset = pd.read_csv("listing.csv")
    cleaned_data = clean_tabular_data(original_dataset)
    cleaned_data.to_csv("clean_tabular_data.csv") #save cleaned dataframe in current directory with given name