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

## Regression model

Now that we have cleaned our database and rescaled our image files, we can begin to create our mathematical models. Our first set of models regards regression which yields continuous outputs. These models can be used to predict outputs with variables associated with continuous value. For our case, we shall use that to predict the property's *Night_Price* value using relevant data.

### SGDRegressor

Stochastic Gradient Descent Regression is an algorithm used to find the optimal linear model. The initial step requries us to make 

$$
\widetilde{y} = h(x) = x \cdot w_0+b.
$$

Here $w$ represents the weight vector which contains the weights for each input in the parameter vector $x$ and $b$ is the bias value. Then, the Loss function is defined as followed:

$$
L := \frac{1}{M}\sum_{i=0}^{M}(\widetilde{y}_i-y_i)^2.
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

$R^2$ is defined with the range [-1,1]. If our measure is valued at close to 1, then our data has a strong positive correlation between the variables. If our measure is close to -1, then there exists a strong negative correlation. And if our measure is close 0, the no (or a very weak) correlation is present. These measures where used to judge the success of our models in predicting our desired data. For this context, the close our predictions are to the original data, the large $R^2$ will be. This is classically used to determine the line-of-best-fit, however this can be extended to non-linear equations (models). 

### Custom tuning hyperparameters

### GridSearchCV

### Saving model

### Decision trees

### Random forests

### Gradient boosting

### Finding best model with params.
