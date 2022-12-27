from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import torch
import pandas as pd
import numpy as np
import yaml
import ast
import os
import json
import time
import random
from datetime import date #import to get current years, months and days value (eg date.today().year)
from time import strftime #import to get current hour, minute and second (eg strftime('%H'))
from yaml.loader import SafeLoader

def get_nn_config(yaml_file):
    with open(yaml_file) as f:
        config = yaml.load(f, Loader=SafeLoader)
    return config

class AirbnbNightlyPriceImageDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = pd.read_csv('clean_tabular_data.csv')
    
    def __getitem__(self,index):
        row = self.data.iloc[index]
        features = torch.tensor(row[['guests','beds','bathrooms','Cleanliness_rating','Accuracy_rating','Communication_rating','Check-in_rating','Value_rating','amenities_count','bedrooms']])
        label = torch.tensor(row['Price_Night'])
        return (features,label)

    def __len__(self):
        length = len(self.data)
        return length

if __name__ == "__main__":
    dataset = AirbnbNightlyPriceImageDataset()

    train_set, test_set = random_split(dataset,[round(len(dataset)*0.9),round(len(dataset)*0.1)])
    train_set, valid_set = random_split(train_set,[round(len(train_set)*0.8),round(len(train_set)*0.2)])

    train_loader = DataLoader(train_set, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=True)

    features, label = next(iter(train_loader))

class LinearRegression(torch.nn.Module): 
    def __init__(self,nn_config):
        super().__init__()

        hidden_layers = nn_config['hidden_layer_width'] #list of respective width of layer

        self.layers = torch.nn.Sequential( #Sequential function used to create a multi layered neural network
            torch.nn.Linear(10,hidden_layers[0]), #initial layer of neural network mapped to a layer of 15 nodes via a linear transform
            torch.nn.ReLU(), #initialise ReLU activator function
            torch.nn.Linear(hidden_layers[0],hidden_layers[1]), #15 node layer maps to a 5 node layer via linear transform
            torch.nn.ReLU(), #initialise ReLU activator function
            torch.nn.Linear(hidden_layers[1],1) #5 node layer maps to the desired output node via linear transform
        ) # Checked Nrural Network of structure 5, 10 and 15-5

        #self.linear_layer = torch.nn.Linear(10,1)

    def forward(self, features):
        return self.layers(features)

features = features.to(torch.float32) #Loader default sets the outputs to float 64, chnage to float 32 for model to work

def train(model, train_loader, config, epochs=50):
    #start = torch.cuda.Event(enable_timing=True) create instance of the start object
    #end = torch.cuda.Event(enable_timing=True) create instance of the end object
    os.chdir('C:/Users/Daniel H/Desktop/AI Core/Python/Airbnb')
    start_time = time.time()
    #start.record() start recording using the start object
    optimiser_method = config['optimiser']
    optimiser_method = eval(optimiser_method) #to convert string into optimiser method

    learn_rate = config['learning_rate']

    optimiser = optimiser_method(model.parameters(),lr=learn_rate) #BEWARE not to set learning-rate (lr) too high!
    writer = SummaryWriter()
    batch_idx = 0

    for epoch in range(epochs):
        for batch in train_loader:
            features, labels = batch
            features = features.to(torch.float32)
            labels = labels.to(torch.float32).reshape(-1,1)
            prediction = model(features)
            loss = F.mse_loss(prediction, labels)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            writer.add_scalar('loss', loss, batch_idx)
            batch_idx += 1
    #end.record() record end time using the end object
    train_time = time.time() - start_time

    #torch.cuda.synchronize() wait for the process to properly finish
    
    #train_duration = start.elapsed_time(end) calculate time taken by the process
    
    #if __name__ == '__main__':
        #print(f'Time for model to be trained: {train_time} seconds.')

    try:
        os.makedirs('models/regression/neural_networks/')
    except:
        pass
    os.chdir('models/regression/neural_networks/')

    year, month, day = date.today().year, date.today().month, date.today().day #find current year, month, day
    hour, minute, second = strftime('%H'), strftime('%M'), strftime('%S') #find current hour, minute, second value
    if len(str(day))==1: #formatting: if day is single valued... 
        day = "0"+str(day) #concatinate a 0 to the front
    folder_name = f"{year}-{month}-{day}_{hour}-{minute}-{second}" #create folder name to save trained model in
    os.mkdir(folder_name) #create folder with created folder_name
    os.chdir(folder_name) #change working directory to folder
    torch.save(model.state_dict(),'model.pt') #save trained model in folder

    json_string = json.dumps(config)
    parameter_file = open("hyperparameters.json","w")
    parameter_file.write(json_string)
    parameter_file.close()

    os.chdir('C:/Users/Daniel H/Desktop/AI Core/Python/Airbnb') #revert to desired working directory
    return model, train_time #return trained model with train_duration

def valid_and_test_accuracy(model,valid_loader,test_loader): #accuracy cannot be assessed for continuous variable outputs, so a plot will be created to judge the success of the predictive model
    for loader in [valid_loader, test_loader]: #create plots for validation and test sets
        prediction_writer = SummaryWriter() #class instance to allow the data to be mapped to the tensor board for visualisation
        batch_idx = 0 #x-axis for the tensor board plots
        for batch in loader: #for loop for the outputs of the predictive model
            features, labels = batch #unpack the data
            features = features.to(torch.float32) #convert vales to torch.float32
            labels = labels.to(torch.float32).reshape(-1,1) #convert values to torch.float32 and reshape to use with features
            pred = model(features) #run model with given features
            prediction_writer.add_scalar('Predictions/Real_Value', pred, batch_idx) #add value to tensor board datavector corresponding to plot titles 'Predictions/Real_Value' in folder 'runs'
            batch_idx += 1 #add one for next prediction value
            
        data_writer = SummaryWriter() #call another instance so the data is separated into different vectors
        batch_idx = 0 #resetting x-axis vector
        for batch in loader: #for loop for the labels which we are trying to predict
            features, labels = batch #unpack data
            labels = labels.to(torch.float32).reshape(-1,1) #converting to torch.float32 and reshapping
            data_writer.add_scalar('Predictions/Real_Value', labels, batch_idx) #add scalar data of labels to tensorboard
            batch_idx += 1 #add one for next label value
    return

def get_loader_mean(loader):
    scaled_mean_list = []
    loader_length = 0
    for batch in loader:
        features, labels = batch
        labels = labels.to(torch.float32).reshape(-1,1)
        batch_length = len(batch[1])
        labels_mean = torch.mean(labels)
        labels_mean = labels_mean.item()
        scaled_mean_list.append(labels_mean*batch_length)
        loader_length += batch_length
    mean = sum(scaled_mean_list)/loader_length
    return mean

def get_metrics(model,train_loader,test_loader,valid_loader):
    #start = torch.cuda.Event(enable_timing=True) create instance of a 'start' object
    #end = torch.cuda.Event(enable_timing=True) create instance of an 'end' object
    
    prediction_times = []

    RMSE_list = []
    R2_list = []
    SS_res = 0
    SS_tot = 0
    for loader in [train_loader,test_loader,valid_loader]:
        scaled_MSE_list = []
        mean = get_loader_mean(loader)
        for batch in loader:
            features, labels = batch
            features = features.to(torch.float32)
            labels = labels.to(torch.float32).reshape(-1,1)

            start_time = time.time() #fix time at beginning of prediction
            # start.record() #start recording using start object
            prediction = model(features)
            #print(prediction)
            prediction_time = time.time() - start_time #calculate current time to start time (duration of process)
            # end.record() #end record using end object
            #torch.cuda.synchronize() #wait for timed process to finish
            #pred_time = start.elapsed_time(end) #calculate time the process took to finish

            MSE_loss = F.mse_loss(prediction, labels)
            MSE_loss.backward()

            batch_length = len(batch[0])
            scaled_MSE_list.append(batch_length*(MSE_loss.item()))

            SS_res += sum((labels - prediction)**2)
            SS_tot += sum((labels - mean)**2)

            prediction_times.append(prediction_time/batch_length)

        SS_res = SS_res.item()
        SS_tot = SS_tot.item()
        #print('****************')
        #print(f'{SS_res}\n{SS_tot}')

        RMSE = np.sqrt( sum(scaled_MSE_list)/(len(loader)*batch_length) )
        #print(f'{RMSE}')
        RMSE_list.append(RMSE)

        R2 = 1 - (SS_res/SS_tot)
        R2_list.append(R2)
    
        inference_latency = sum(prediction_times)/len(prediction_times) #average inference_latency in seconds
    return {'training_set': {'RMSE': RMSE_list[0], 'R2': R2_list[0]}, 'test_set': {'RMSE': RMSE_list[1], 'R2': R2_list[0]}, 'validation_set': {'RMSE': RMSE_list[2], 'R2': R2_list[2]}, 'inference_latency': inference_latency}

if __name__ == '__main__':
    os.chdir('C:/Users/Daniel H/Desktop/AI Core/Python/Airbnb/')
    nn_config = get_nn_config('nn_config.yaml')
    model = LinearRegression(nn_config)
    dict = get_metrics(model,train_loader,test_loader,valid_loader)
    print(dict)

def save_model(save_folder_path,model,train_loader,test_loader,valid_loader,nn_config):
    model, train_duration = train(model,train_loader,nn_config)
    
    os.chdir('C:/Users/Daniel H/Desktop/AI Core/Python/Airbnb/neural_networks')
    try:
        os.makedirs(save_folder_path)
    except:
        pass
    
    os.chdir('C:/Users/Daniel H/Desktop/AI Core/Python/Airbnb/neural_networks/'+save_folder_path)   
    torch.save(model.state_dict(),'model.pt') #save model.pt in current directory
    
    json_string = json.dumps(nn_config)
    parameter_file = open("hyperparameters.json","w")
    parameter_file.write(json_string)
    parameter_file.close()

    metrics = get_metrics(model,train_loader,test_loader,valid_loader)
    metrics['training_duration'] = train_duration

    json_string = json.dumps(metrics)
    metric_file = open("metrics.json","w")
    metric_file.write(json_string)
    metric_file.close()
    return

if __name__ == "__main__":
    os.chdir('C:/Users/Daniel H/Desktop/AI Core/Python/Airbnb')
    nn_config = get_nn_config('nn_config.yaml')
    model = LinearRegression(nn_config)
    save_model('regression',model,train_loader,test_loader,valid_loader,nn_config)

def generate_nn_configs(nn_config):
    nn_config_list = []
    learning_rates = [0.01,0.005,0.001]
    for lr in learning_rates:
        for i in range(6): #want 16 permutations
            layer1 = random.randint(10,15)
            layer2 = random.randint(4,9)
            new_nn_config = nn_config.copy()
            new_nn_config['learning_rate'] = lr
            new_nn_config['hidden_layer_width'] = [layer1,layer2]
            nn_config_list.append(new_nn_config)
    return nn_config_list

def find_best_nn(train_loader, test_loader, valid_loader, nn_config_list):
    os.chdir('C:/Users/Daniel H/Desktop/AI Core/Python/Airbnb/')
    RMSE_list = []
    for config in nn_config_list:
        untrained_model = LinearRegression(config)
        trained_model, train_time = train(untrained_model, train_loader, config)
        metrics = get_metrics(trained_model,train_loader,test_loader,valid_loader)
        metrics['train_duration'] = train_time
        print('***********************************************************************************************')
        print(f'{config} \n')
        print(metrics)
        #print(type(metrics['test_set']['RMSE'].item()))
        if str(metrics['test_set']['R2']) != 'nan':
            RMSE_list.append(metrics['test_set']['R2'])
            if metrics['test_set']['R2'] == max(RMSE_list): #determining best model by RMSE score of the test set
                best_model = trained_model
                best_model_metrics = metrics
                best_config = config
    
    try:
        os.makedirs('models/regression/neural_networks/best_neural_network')
    except:
        pass
    os.chdir('C:/Users/Daniel H/Desktop/AI Core/Python/Airbnb/models/regression/neural_networks/best_neural_network/')
    
    torch.save(best_model.state_dict(),'model.pt') #save model.pt in current directory
    
    json_string = json.dumps(best_config)
    parameter_file = open("hyperparameters.json","w")
    parameter_file.write(json_string)
    parameter_file.close()

    json_string = json.dumps(best_model_metrics)
    metric_file = open("metrics.json","w")
    metric_file.write(json_string)
    metric_file.close()

    os.chdir('C:/Users/Daniel H/Desktop/AI Core/Python/Airbnb/')
    return best_model, best_model_metrics, best_config

if __name__ == '__main__':
    nn_config_list = generate_nn_configs(nn_config)
    best_model, best_model_metrics, best_config = find_best_nn(train_loader, test_loader, valid_loader, nn_config_list)
    print('*****************************************************')
    print(f'Best Model: \n {best_model} \n')
    print(f'Best Model Metrics: \n {best_model_metrics} \n')
    print(f'Best Model config: \n {best_config} \n')