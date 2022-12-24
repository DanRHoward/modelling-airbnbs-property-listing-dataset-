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

## Neural Network

![Alt text](file:///C:/Users/Daniel%20H/Downloads/nn.svg)
<img src="file:///C:/Users/Daniel%20H/Downloads/nn.svg">

![nn](https://user-images.githubusercontent.com/116043233/209418479-7d670577-617d-4e42-9a70-d09059a48fbb.svg)

![NeuralNetworkDiagram](https://user-images.githubusercontent.com/116043233/209418621-afeee6a9-bab4-46b2-bb10-1bb033c09ee2.png)

