from tabular_data import load_airbnb
from modelling import save_model
import pandas as pd
from sklearn.linear_model import LogisticRegression #import logistic regression model
from sklearn.model_selection import train_test_split #import function to randomly split ur data into train and test sets
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import os
import json
import joblib

def my_LogisticRegression():
    clean_dataset = pd.read_csv("clean_tabular_data.csv")
    tuple = load_airbnb(clean_dataset,'Category')

    X = clean_dataset[['guests','beds','bathrooms','Price_Night','Cleanliness_rating','Accuracy_rating','Communication_rating','Check-in_rating','Value_rating','amenities_count','bedrooms']].values  #all data that suggests to our target data (all numerical data that is not the data we want to predict)
    y = tuple[0] #the data we want to predict (5 types of proerty to predict)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42) #use function to define our train and test sets with 

    model = LogisticRegression(max_iter = 10000) #max_iter defined to avoid warning message when run by default
    model.fit(X_train,y_train)

    prediction = model.predict(X_test)

    cf = confusion_matrix(prediction,y_test)
    print(f'Confusion matrix, \n {cf} \n')

    precision_value = precision_score(prediction,y_test,average='weighted',zero_division=0)
    print(f'Precision Score: {precision_value} \n')

    recall_value = recall_score(prediction,y_test,average='weighted')
    print(f'Recall Score: {recall_value} \n')

    f1_value = f1_score(prediction,y_test,average='weighted')
    print(f'F1 Score: {f1_value} \n')

    model_score = model.score(X_test,y_test)
    print(f'Mean Accuracy score: {model_score}') #Mean Accuracy score, in interval [0,1] (In this case, roughly 42% correctly predicted. Not too bad considering we have 5 different types of property to predict)
    return 

def tune_classification_model_hyperparameters(model_class,train_sets,test_sets,validation_sets,hyper_dict):
    try:
        gridsearch = GridSearchCV(model_class(max_iter=100000), hyper_dict)
    except:
        gridsearch = GridSearchCV(model_class(),hyper_dict)
    
    fit_model = gridsearch.fit(train_sets[0],train_sets[1])
    best_estimator = fit_model.best_estimator_

    prediction = best_estimator.predict(test_sets[0])

    precision_value = precision_score(prediction,test_sets[1],average='weighted',zero_division=0)
    recall_value = recall_score(prediction,test_sets[1],average='weighted')
    f1_value = f1_score(prediction,test_sets[1],average='weighted')
    accuracy_score = best_estimator.score(test_sets[0],test_sets[1])

    validation_accuracy_score = best_estimator.score(validation_sets[0],validation_sets[1])
    
    metrics = {
        "Validation Accuracy Score": validation_accuracy_score, 
        "Precision Score": precision_value,
        "Recall Score": recall_value,
        "F1 Score": f1_value,
        "Mean Accuracy Score": accuracy_score
    }
    return [model_class, gridsearch.best_params_, metrics]

def evaluate_all_models(train_sets,test_sets,validation_sets):
    decisiontree_hyperparameters = {
        'criterion': ['gini','entropy','log_loss'],
        'splitter': ['best','random'],
        'max_features': [None,'auto','sqrt','log2']
    }
    randomforest_hyperparameters = {
        'criterion': ['gini','entropy','log_loss'],
        'min_samples_leaf': [1,2,3],
        'max_features': ['sqrt','log2',None]
    }
    gradientboosting_hyperparameters = {
        'loss': ['log_loss','deviance','exponential'],
        'criterion': ['friedman_use','squared_error'],
        'max_features': ['auto','sqrt','log2']
    }
    model_hyperparameters_list = [decisiontree_hyperparameters,randomforest_hyperparameters,gradientboosting_hyperparameters]
    model_list = [DecisionTreeClassifier,RandomForestClassifier,GradientBoostingClassifier]
    file_name_list = ['desicion_tree','random_forest','gradient_boost']
    os.chdir('C:/Users/Daniel H/Desktop/AI Core/Python/Airbnb/')
    for (model, params, file_name) in zip(model_list, model_hyperparameters_list, file_name_list):
        try:
            os.makedirs('models/classification/'+ file_name)
        except:
            pass
        save_model(tune_classification_model_hyperparameters,f'models/classification/{file_name}',model,train_sets,test_sets,validation_sets,params)
        os.chdir('C:/Users/Daniel H/Desktop/AI Core/Python/Airbnb/')
    return

if __name__ == "__main__":
    os.chdir('C:/Users/Daniel H/Desktop/AI Core/Python/Airbnb/')
    clean_dataset = pd.read_csv('clean_tabular_data.csv') #Load 
    tuple = load_airbnb(clean_dataset,'Category')
    
    X = clean_dataset[['guests','beds','bathrooms','Price_Night','Cleanliness_rating','Accuracy_rating','Communication_rating','Check-in_rating','Value_rating','amenities_count','bedrooms']].values  #all data that suggests to our target data (all numerical data that is not the data we want to predict)
    y = tuple[0] #the data we want to predict (5 types of proerty to predict)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25)

    train_sets, test_sets, validation_sets = (X_train,y_train), (X_test,y_test), (X_valid,y_valid)
    evaluate_all_models(train_sets,test_sets,validation_sets)

def find_best_model():
    working_folder_path = "C:/Users/Daniel H/Desktop/AI Core/Python/Airbnb/models/classification"
    os.chdir(working_folder_path)
    VAS_scores = []

    for dir in os.listdir(working_folder_path): #for loop to gather all Validation Accuracy Scores for our best models for each class
        os.chdir(dir)
        with open("metrics.json") as jsonFile:
            jsonObject = json.load(jsonFile)
            jsonFile.close()
        
        VAS_scores.append(jsonObject['Validation Accuracy Score'])
        os.chdir(working_folder_path)

    best_VAS_score = max(VAS_scores) #finding best Validation Accuracy score
    
    for dir in os.listdir(working_folder_path):
        os.chdir(dir)
        with open("metrics.json") as jsonFile:
            jsonObject = json.load(jsonFile)
            jsonFile.close()
        if jsonObject['Validation Accuracy Score'] == best_VAS_score:
            best_model_metrics = jsonObject
            with open("hyperparameters.json") as jsonFile:
                jsonObject = json.load(jsonFile)
                jsonFile.close()
            best_model_hyperparameters = jsonObject
            loaded_model = joblib.load("model")
        os.chdir(working_folder_path)
    return loaded_model, best_model_metrics, best_model_hyperparameters

if __name__ == "__main__":
    best_model = find_best_model()
    print(best_model)