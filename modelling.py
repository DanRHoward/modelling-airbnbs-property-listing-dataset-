import pandas as pd
import numpy as np
from tabular_data import load_airbnb
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.preprocessing import scale
import sklearn
import os
import json
import joblib

def my_regressor_model(model_class): #Help: https://pythonprogramming.net/machine-learning-python3-pandas-data-analysis/
    os.chdir('C:/Users/Daniel H/Desktop/AI Core/Python/Airbnb/')
    clean_tabular_data = pd.read_csv('clean_tabular_data.csv') #Load 
    tuple = load_airbnb(clean_tabular_data,'Price_Night')
    
    X = clean_tabular_data[['guests','beds','bathrooms','Price_Night','Cleanliness_rating','Accuracy_rating','Communication_rating','Check-in_rating','Value_rating','amenities_count','bedrooms']].drop(tuple[1], axis = 1).values  #all data that suggests to our target data (all numerical data that is not the data we want to predict)
    X = scale(X) #converts data into relative interval in [0,1], if not used prediction becomes unreliable
    y = tuple[0] #the data we want to predict
    
    test_size = 100 #number of data points to use to create our model

    X_train = X[:-test_size] #take data up to the last 100 rows to train our model
    y_train = y[:-test_size] #take data up to the last 100 rows to train our model

    X_test = X[-test_size:] #for the last 100 rows to test our model
    y_test = y[-test_size:] #for the last 100 rows to test our model

    sgdr = model_class() #creates an instance of the desired model class
    fit = sgdr.fit(X_train,y_train) #find the best fitting model for our training data
    prediction = fit.predict(X_test) #predict using the test data what 'y' (Price_Night) would you expect to be

    #for X,y in zip(X_test, y_test): #for X and y entries in respective lists...
    #    print(f"Model: {int(fit.predict([X])[0])}. Actual: {y}") #print. Note that even though we are entering onyl one point in to the predict function, it returns a list so just take the first entry of that list
    
    R_squared = sgdr.score(X_test,y_test) # R^2 score of the data sets
    MSE = mean_squared_error(y_test, prediction) #evaluates the Mean-Squared-Error
    RMSE = MSE**0.5 #evaluates the Root-Mean-Squared-Error

    #print(f"\nR-squared: {R_squared}") #print R^2 value
    #print(f"RMSE: {RMSE}") #prints the Root-Mean-Squared-Error
    return ({'R_squared': R_squared, 'RMSE': RMSE}, sgdr) #returns a dictionary of the models metrics and the instance of the model to be saved

def tune_regression_model_hyperparameters(model,params_grid):
    os.chdir('C:/Users/Daniel H/Desktop/AI Core/Python/Airbnb/')
    clean_tabular_data = pd.read_csv('clean_tabular_data.csv') #Load 
    tuple = load_airbnb(clean_tabular_data,'Price_Night')
    
    X = clean_tabular_data[['guests','beds','bathrooms','Price_Night','Cleanliness_rating','Accuracy_rating','Communication_rating','Check-in_rating','Value_rating','amenities_count','bedrooms']].drop(tuple[1], axis = 1).values  #all data that suggests to our target data (all numerical data that is not the data we want to predict)
    X = scale(X) #converts data into relative interval in [0,1], if not used prediction becomes unreliable
    y = tuple[0] #the data we want to predict
    
    test_size = 100 #number of data points to use to create our model

    X_train = X[:-test_size] #take data up to the last 100 rows
    y_train = y[:-test_size] #take data up to the last 100 rows

    X_test = X[-test_size:] #for the last 100 rows to test our model
    y_test = y[-test_size:] #for the last 100 rows to test our model
    
    try:
        gridsearch = sklearn.model_selection.GridSearchCV(model(max_iter=100000),params_grid)
    except:
        gridsearch = sklearn.model_selection.GridSearchCV(model(),params_grid)

    fit=gridsearch.fit(X_train,y_train)
    best_estimator = fit.best_estimator_
    predict = best_estimator.predict(X_test)
    R_squared = fit.score(X_test,y_test)
    RMSE = mean_squared_error(y_test,predict)**0.5
    #fit = gridsearch.fit(X_train,y_train)
    #predict = fit.predict(X_test)
    #RMSE = mean_squared_error(y_test,predict)**0.5
    #print(RMSE)
    #print(f"The most optimal model is;\n\n{gridsearch.best_params_}")
    return [model, gridsearch.best_params_, {'R_squared': R_squared, 'RMSE': RMSE}]

def save_model(tune_model_function,save_folder_path,model,train_sets,test_sets,validation_sets,params_grid):
    os.chdir('C:/Users/Daniel H/Desktop/AI Core/Python/Airbnb/')
    try:
        os.makedirs(save_folder_path)
    except:
        pass
    tuned_hyperparameters = tune_model_function(model,train_sets,test_sets,validation_sets,params_grid)
    #SGDRegressor_model_output = my_regressor_model()
    model = tuned_hyperparameters[0]
    parameters = tuned_hyperparameters[1]
    metrics = tuned_hyperparameters[2]
    
    os.chdir('C:/Users/Daniel H/Desktop/AI Core/Python/Airbnb/'+save_folder_path)   
    joblib.dump(model,"model")

    json_string = json.dumps(parameters)
    parameter_file = open("hyperparameters.json","w")
    parameter_file.write(json_string)
    parameter_file.close()

    json_string = json.dumps(metrics)
    metric_file = open("metrics.json","w")
    metric_file.write(json_string)
    metric_file.close()
    return

def evaluate_all_models():
    SGDRegressor_hyperparameters = {
            'loss': ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
            'penalty': ['l2','l1','elasticnet'],
            'alpha': [1e-3,1e-4,1e-5]
        }
    decisiontree_hyperparameters = {
        'criterion': ['squared_error','friedman_mse','absolute_error','poisson'],
        'splitter': ['best','random'],
        'max_features': ['auto','sqrt','log2']
    }
    randomforest_hyperparameters = {
        'criterion': ['squared_error','absolute_error','poisson'],
        'min_weight_fraction_leaf': [0.0,1e-5,1e-3],
        'max_features': ['sqrt','log2',None]
    }
    gradientboosting_hyperparameters = {
        'loss': ['squared_error','absolute_error','huber','quantile'],
        'min_weight_fraction_leaf': [0.0,1e-5,1e-3],
        'max_features': ['auto','sqrt','log2']
    }
    model_hyperparameters_list = [SGDRegressor_hyperparameters,decisiontree_hyperparameters,randomforest_hyperparameters,gradientboosting_hyperparameters]
    model_list = [SGDRegressor,DecisionTreeRegressor,RandomForestRegressor,GradientBoostingRegressor]
    file_name_list = ['linear_regression','desicion_tree','random_forest','gradient_boost']
    os.chdir('C:/Users/Daniel H/Desktop/AI Core/Python/Airbnb/')
    for (model, params, file_name) in zip(model_list, model_hyperparameters_list, file_name_list):
        try:
            os.makedirs('models/regression/'+ file_name)
        except:
            pass
        save_model(f'models/regression/{file_name}',model,params)
        os.chdir('C:/Users/Daniel H/Desktop/AI Core/Python/Airbnb/')
    return

if __name__ == "__main__":
    evaluate_all_models()