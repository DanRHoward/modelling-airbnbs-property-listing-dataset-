# modelling-airbnbs-property-listing-dataset-

During this project, we will be handling a given dataset, creating an appropriate model and evaluating the success of that model in predicting the desired infomation. When dealing with data modelling, there are 4 stages to consider;

- **Data cleanning and handling**. During this step, the data set is cleaned adn prepared for it's use in creating our model. This can include removing corrupted data or reformatting existing data to allow it to be used correctly.

- **Creating a appropriate model**. During this step, the initial model is created.

- **Evaluating the model's criterion**. During this step, we evaluate the model's success at predicting the desired infomation. This can be evaluated using using certain metrics such as RMSE and $R^2$.

- **Optimise the model**. This final step requires us to optimise our model. This can be achieved by cycling through the model's parameters and calculate the previously mentioned criterion metrics to find the best performing model. This can be done using already built functions such as GridSearchCV.

## Data Preparation

### Data cleaning

### Image rescaling

### Appropriate storage

## Regression model

### SGDRegressor

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
