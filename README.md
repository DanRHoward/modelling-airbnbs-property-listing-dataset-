# modelling-airbnbs-property-listing-dataset-

During this project, we will be handling a given dataset, creating an appropriate model and evaluating the success of that model in predicting the desired infomation. When dealing with data modelling, there are 4 stages to consider;

- **Data cleanning and handling**. During this step, the data set is cleaned adn prepared for it's use in creating our model. This can include removing corrupted data or reformatting existing data to allow it to be used correctly.

- **Creating a appropriate model**. During this step, the initial model is created.

- **Evaluating the model's criterion**. During this step, we evaluate the model's success at predicting the desired infomation. This can be evaluated using using certain metrics such as RMSE and $R^2$.

- **Optimise the model**. This final step requires us to optimise our model. This can be achieved by cycling through the model's parameters and calculate the previously mentioned criterion metrics to find the best performing model. This can be done using already built functions such as GridSearchCV.

## Data Preparation

The first step during this project was to perform various data cleaning processes to the given .csv file which contains our relevant data. The given initial database can be found in *listings.csv*.

### Data cleaning

Data cleaning is the process of detecting and correcting (or removing) corrupt or inaccurate records from a record set, table, or database and refers to identifying incomplete, incorrect, inaccurate or irrelevant parts of the data and then replacing, modifying, or deleting the dirty or coarse data. This is to allow us to use the data effectively in creating reliable mathematical models which produce unbias predictions. For our case, rows of data were to be removed due to lack of data in certain columns, shifting column data across due to an error distorting the databases structure and by setting values for certain missing data which can be safely assumed.

The code for this can be found in *tabular_data.py*. The cleaned database can be found in *clean_tabular_data.csv*.

### Image rescaling

Along with data cleansing our database, we were also instructed to reformat the image files that were given along with our .csv file. A folder was given which contained folder for each of the property within our database. Within these property files contains several image files of the respective property. With these image files, we were instructed to rescale each of these so that the height of ever image file is equal to the minimum height of the present images and that every image preserved their aspect ratio. This was achieved by simply setting each width to the minimum, $\min(\text{width})$, and the height is multiplied by $\frac{\min(\text{width})}{\text{width}}$.

The code for this can be found in *prepare_image_data.py*.

## Regression models

Now that we have cleaned our database and rescaled our image files, we can begin to create our mathematical models. Our first set of models regards regression which yields continuous outputs. These models can be used to predict outputs with variables associated with continuous value. For our case, we shall use that to predict the property's *Night_Price* value using relevant data.

### SGDRegressor

Stochastic Gradient Descent Regression is an algorithm used to find the optimal linear model. The initial step requries us to make 

$$
\widetilde{y} = h(x) = x \cdot w_0+b.
$$

Here $w$ represents the weight vector which contains the weights for each input in the parameter vector $x$ and $b$ is the bias value. Then, the Loss function is defined as followed:

$$
L := \frac{1}{N}\sum_{i=0}^{N-1}(\widetilde{y}_i-y_i)^2.
$$

This function defines an error value for the difference between our approximation and the real values we have from our data set. From here, we evaluate the new weight vector for an approximation using this loss function by:

$$
w_{i+1} := w_i + \eta \nabla L.
$$

$\eta$ denotes the learning rate of the algorithm, the larger it is, the larger steps size it takes towards the convergent value (Beware of too large a value, as it may miss the convergent value). This new weight vector is then substituted into the approximation function $\widetilde{y}$ to get a new model function. We then repeat these steps until our weight vector converges to an optimal weight vector for our parameters,

$$
w_i \longrightarrow{} w^*.
$$

Using this, we can create our model as $h(x) = w^* \cdot x + b$. We were not tasked with recreating this process from scratch, rather we utilised the already built process with the *sklearn* package and method *SGDRegressor*. Using this, we were able to derive predictions with our test sets for the *Price_Night* of given property labels.

### RMSE and $R^2$ measures

The two metrics that we measured our current model and future models were the RMSE and the $R^2$ value. RMSE stands for the Root-Mean-Squared-Error (or Root-Mean-Squared-Deviation) is a common measure of the distance bewteen an output from a prodictive model and the actual value it intends to model. Thus, the measure is always valued as a non-negative real number. If our predicted data is denoted as $\widetilde{y}_i$ and our actual data is denoted as $y_i$, then the measure is defined as;

$$
RMSE := \sqrt{\sum_{i=1}^{N}\frac{(\widetilde{y}_i-y_i)^2}{N}},
$$

with $N$ being the number of data points previded. 

$R^2$ is known as the *Coefficient of Determination*. Similar to RMSE, $R^2$ can be used as a measure of the distribution of data point from our real-world observation. However, this can also be used to evaluate the correlation-coefficient to determine if there is a strong or weak correlation between the involved variables. This measure can be evaluated using;

$$
R^2 := 1 - \frac{\sum (\widetilde{y}_i - y_i)^2}{\sum (\bar{y}-y_i)^2},
$$

with $\bar{y}$ referring to the mean of the actual data, defined as;

$$
\bar{y} := \frac{\sum y_i}{N}.
$$

$R^2$ is defined with the range [0,1]. If our measure is valued at close to 1, then our data has a strong positive correlation between the variables. If our measure is close 0, the no (or a very weak) correlation is present. The possability of negative values being produced are not nonexistant. If a negative value is roduces, it means that the model being evaluated is performing worse than simply taking the average of the given data. The extent of how worse the model is at prediction compared to the average is determined by the magnitude of the $R^2$ value. These measures where used to judge the success of our models in predicting our desired data. For this context, the close our predictions are to the original data, the large $R^2$ will be. This is classically used to determine the line-of-best-fit, however this can be extended to non-linear equations (models). 

### Custom tuning hyperparameters

The next task required the creation of a hyperparameter tuning process from scratch. This process would cycle through ever permutation of each parameter of a given modelling method. The model's performance metrics would then be evaluated and compared to other mdoels during this process. The best performing model, with it's hyperparameters values, would then be return. Since every problem is inherently unique in naturely, not every set of parameters for certain models will be optimal for every problem. Thus, we must perfrom this process in order to find the *best* model for our situation. In a work scenario, a from-scratch approach to this is not necessary. 

### GridSearchCV

*sklearn* package contains an already built function to perfrom tuning hyperparameters. This function is called *GridSearchCV*. We can call this function instead creating our own process. Within python file *modelling.py*, we can find this function in use within the *tune_regression_model_hyperparameters*.

### Decision trees

We have previously mentioned the using Stochastic Gradient Descent (SGD) modelling for creating predictive models for our data. However, this isn't the only approach we could take when modelling. Another method involves the use of *decision trees*. This method involves breaking down the dataset into smaller and smaller subsetsat the same time and associates each using characteristics of the data within the data subsets. This is typically done by observing the values of the data and noticing given patterns. For example, if we want to predict the salary of a given person and every person that is an employee of a given company is given a high salary, this pattern will be used in the predictive model explicitely. The final result is a tree-like structure with decision nodes and leaf nodes, much like if an algorithm was written out using pen and paper with boxes of actions and arrows for chocen decisions. The leaf nodes represent the outsomes of the decision tree.

Further infomation about Decision Tree Regression modelling can be found here:

https://towardsdatascience.com/machine-learning-basics-decision-tree-regression-1d73ea003fda

### Random forests

*Random forest* modelling uses the same method structure and logic behind desicion trees, except this process is multiplied to created a *forest*. One method that this is achieved is to partition or generate multiple subsets of the main dataset. A decision tree is then created for each of this sub-divisions. The outcomes of each of the decision trees is then evaluated and is used to evaluate the final output of the model. For instance, if the classification version of the model is used, then the final output of the forest model is determined by the majority decision from every decision tree created.

Further infomation regarding this model can be found:

https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/

### Gradient boosting

*Gradient boosting* is another method of modelling which produces a sequence of models whose predictions are summed together to produce a final prediction. Specifically, the models which are produced predict the pseudo residuals between the previous model's and the the ideal output. The initial model produced is a simple constant function of the form;

$$
h_0(x) = c,
$$

where $x$ is the input data and $c$ is some constant. Suppose that we want to find what is the *best* value for $c$, then it is natural fo us to assume that this value would be for what value of $c$ would yield us the smallest MSE between $c$ and our data $y$. This value can now be calculated as follows;

$$
\begin{split}
   c^* & = \underset{c}{\text{argmin}}\left( \text{MSE}(c,y)\right),\\
   & = \underset{c}{\text{argmin}}\left( \frac{1}{N} \sum_{i=1}^{N}(c-y_i)^2 \right),\\
   \implies 0 & = \nabla_c \frac{1}{N} \sum_{i=1}^{N}(c^* - y_i)^2,\\
   & = \frac{2}{N} \sum_{i=1}^{N} (c^* - y_i),\\
   & = 2c^* - \frac{2}{N} \sum_{i_1}^{N} y_i,\\
   \implies c^* & = \frac{1}{N} \sum_{i=1}^{N} y_i =: \mathbb{E}[y].
\end{split}
$$

Thus, we can set the initial prediction model $h_0(x)$ to the expectation of the dataset $\bar{y}$. 

$$
h_0(x) = \mathbb{E}[y] := \bar{y} = c^*.
$$

After this, more models are preduced with attempt the residuals of our model that we previously mentioned. We shall denote these models as $h_i$. These models are then sclaed by some learning rate constant that we shall denote as $\alpha$ and summed to the previously predictive model. This takes the form;

$$
\begin{split}
   H_l(x) & = h_0(x) + \alpha h_1(x) + \dots + \alpha h_l(x),\\
   & = h_0(x) + \alpha \sum_{i=1}^{l}h_i(x).
\end{split}
$$

If the learning rate is appropriatly chosen, then as $l$ increases (the more residual predictor models, $h_i$) are added, the smaller the MSE will be and thus the better the predictive model will be.

### Finding best model with params.

Within this project, we were tasked with finding the "*best*" performing model. To achieve this dictionaries for each of the modelling methods was created which contained lists of options for pre-determined parameters. The code would then cycle through each possible permutation of parameters for each modelling method and evaluate the models performance metrics. For our problem, the success of each model was evaluated using the RMSE of the model and the best performing model determined by the smallest value produced by them all. 

The code relating to regression modelling and the evaluation of the best model can be found in the python file: *modelling.py*.

## Classification models

Every model that we have previously mentioned is in regards to regression modelling, that is if the values that we wish to predict are continuous, such as a person's salary or their hieght we want to model. However, some values are not of this nature. Boolean values which return a True or False value or the number of siblings a person has cannot be predicted effectively using regression. Instead, we use Classification modelling to produce models which specialise in dealling with this type of problem.  

For our project, the classification models' purpose was to evaluate the type of property that a model is given.

### Logistic model

The Logistic model takes in data features to evaluate certain parameters of the logistic function. The logistic function takes the form;

$$
p(x) = \frac{1}{1+e^{-(x-\mu)/s}},
$$

where $\mu$ is a location parameter where $p(\mu) = \frac{1}{2}$, and $s$ is a scale parameter. This equation spesifically is applied to a *Binary Logistic* model. This means that the model will only output two possible outcomes. The curve itself can be interpreted as the probability of one of the outcome events occuring. In our project, our model took the form of a *Multinomial Logistic* model. This refers to the fact that there was more that two possible outcomes for our model. The final model type is names the *Ordinal Logistic* model which is the same as the multinormial variant, except that an ordering must take place, ie if we are categorising restaurant rankings which naturally have an ordering behaviour and that predictions cannot have multiple data prediction ranks of the same value.

### Metrics

- Validation Accuracy Score:
- Precision Score:
- Recall Score:
- F1 Score:
- Mean Accuracy Score:

- Confusion Matrix:

## Neural Network



### Simple search for the best neural network

Once an understanding of implementing a neural network using *pytroch* with our project, a question must be asked; "*What structure will our neural network take?*". When first working with neural networks, you discover that the more hidden layers a network has does **NOT** guarantee a better predictive model. Furthermore, it is very difficult to tell whether a given neural network will work for a given problem. Since this was my first experience working with nerual networks, thus was unaware of the subtilties of neural network construction and the inexperience of using any for of evolutionary neural network to optimise the structure itself, I conducted a small search with only three options for the structure. These three options for the neural network structure to take are that the hidden layers consisted of the number of nodes that equalled;
   
   1. 5,
   2. 10,
   3. 15-to-5.

Note that these are the number of nodes in the *hidden* layers, since the input layer and output layer remains constant throught the project (10 and 1 respectively). To evaluate the best model of the given options, each were subjected to optimisation, each with three different learning rates; $lr = 0.1$, $0.01$ and $0.001$. The optimisation process was then plotted beside each other to discover the smallest loss value at the end of optimisation. The following plots are the comparison between the different learning rates for each neural network configuration.

#### Neural Network of hidden layer, 5

![NN5_comparison  lr=0 1](https://user-images.githubusercontent.com/116043233/209706986-96cb3912-ee84-4e4d-8795-f38e6eecc89d.png)

The best learning rate for this structure is $lr = 0.1$.

#### Neural Network of hidden layer, 10

![NN10_comparison  lr=0 1](https://user-images.githubusercontent.com/116043233/209707018-c82cfeda-f10b-4d34-84d2-24e0c8fe61e3.png)

The best learning rate for this structure is $lr = 0.1$.

#### Neural Network of hidden layers, 15-5

![NN15-5_comparison  lr=0 01](https://user-images.githubusercontent.com/116043233/209707035-f51dc297-b3bb-40cf-9878-a6935f848523.png)

The best learning rate for this structure is $lr = 0.01$.

With these observation, we can plot the best performing models from each structure against eachother to find the best model overall. The following plot presents this.

![Best_loss_comparison  NN15-5, lr=0 01](https://user-images.githubusercontent.com/116043233/209707055-99569696-3d99-4d15-a32e-543f28f75171.png)

The best model after optimisation is of hidden layers 15-5 and a learning rate of $lr = 0.01$.

However, in order to evaluate the best models from these plots, the loss time-series data for all the models had to been smoothened since the true values of loss for each iteration of the optimisation process produced incredibly eratic values. It would have been impossible to get any clear insight using the true values of loss, so the smoothened data was used to measure any trending behaviour of the loss curves and to evaluate the best models. Even with this, differences can be incredibly marginal. After this, the bewst model was used to predict the values from our validation and test sets to determine the accuracy of the preedictions. These plots are presented below.

#### Validation set prediction vs observed data

Observed data is represented with the dark blue curve and predicted values are represented with the pale blue.

![Prediction-Labels(Validation_set)](https://user-images.githubusercontent.com/116043233/209707083-c72324a6-cb0b-4232-a931-119c39574b0f.png)

#### Test set prediction vs observed data

observed data is represented with the dark red curve and predicted values are represented with the pale red.

![Prediction-Labels(Test_set)](https://user-images.githubusercontent.com/116043233/209707104-5b7d398c-7dcf-4793-bd28-fbef4551afe4.png)

Notice that the predictions produced are of a constant value, specifically $155.5$. This may suggested that the features of a given property (or atleast the features used for this project) have no impact upon the pricing of the property. Our model has concluded that the best course of action when predicting the Night price of a given property is to simply assume it is equal to $155.5$. This idea is supported by the plot for the validation set found above, as this constant prediction approach is relatively safe, especially if there is indeed no correlation between the features of a property and the price night variable.

#### Expanded search for best neural network

After a restricted search for the best neural network setup was conducted, we were then instructed to perform a more extensive search. A function was created named *generate_nn_configs* in file *neural_network.py*. This function would generate 18 randomly generated neural network configuration which would then be used to construct a neural network and then trained. This trained model would then have it's metrics evaluated and the best model was determined by its best performance metrics. The best model was discovered to be of sturcture 11-7. A diagram that illustrates the neural network is presented below. The colour of the arcs bewtween nodes signifies the value of the weights for each transformation. Blue signifies values close to $1$, where as red signifies value close to $-1$.

![NeuralNetworkDiagram](https://user-images.githubusercontent.com/116043233/209418621-afeee6a9-bab4-46b2-bb10-1bb033c09ee2.png)

The diagram was created using: https://alexlenail.me/NN-SVG/

Even with this optimisation process, result metrics do not suggest that there exists any correlation between property features and the price night variable value.
