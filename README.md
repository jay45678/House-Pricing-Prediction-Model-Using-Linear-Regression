# House-Pricing-Prediction-Model-Using-Linear-Regression

In this you will learn how to create Machine Learning Linear Regression Model. You will be analyzing a house price predication dataset for finding out price of house on different parameters. You will do Exploratory Data Analysis, split the training and testing data, Model Evaluation and Predictions. 

### Problem Statement
A real state agents want the help to predict the house price for regions in the USA. He gave you the dataset to work on and you decided to use Linear Regressioon Model. Create a model which will help him to estimate of what the house would sell for.

Dataset contains 7 columns and 5000 rows with CSV extension. The data contains the following columns :

- 'Avg. Area Income': Avg. Income of householder of the city house is located in.
- 'Avg. Area House Age': Avg. Age of Houses in same city.
- 'Avg. Area Number of Rooms': Avg. Number of Rooms for Houses in same city.
- 'Avg. Area Number of Bedrooms': Avg. Number of Bedrooms for Houses in same city.
- 'Area Population': Population of city.
- 'Price': Price that the house sold at.
- 'Address': Address of the houses.

# An Example: Predicting house prices with linear regression using SciKit-Learn, Pandas, Seaborn and NumPy

### Import Libraries
Install the required libraries and setup for the environment for the project. We will be importing SciKit-Learn, Pandas, Seaborn, Matplotlib and Numpy.
The purpose of “%matplotlib inline” is to add plots to your Jupyter notebook.

### Importing Data and Checking out
As data is in the CSV file, we will read the CSV using pandas read_csv function and check the first 5 rows of the data frame using head().
To know the raw information or to any null value exists or not, we use info() funaction.
For knowing the values like minimum, maximum, median, average, mean, standard deviation and count of houses, we use describe() function.
Columns of the dataset table i.e. dependent(X), independent(y) and other variables which are not used in prediction, all are known by using columns

### Exploratory Data Analysis for House Price Prediction
We will create some simple plot for visualizing the data.
Plots like pairplot and displot for the given data frame to identify the variable variable  between relationship.
sns.pairplot(HouseDF) : This plot is for all the variable variable between relationship.

sns.distplot(HouseDF['Price']) : This plot is for relationship between count and price.

Plot like heatmap of this data frame to identify the dependent and independent variables from other variables. 
sns.heatmap(HouseDF.corr(), annot=True) : This plot is for relationship between variables like 'Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population'and 'Price'.


### Get Data Ready For Training a Linear Regression Model
Let’s now begin to train out the regression model. We will need to first split up our data into an X list that contains the features to train on, and a y list with the target variable, in this case, the Price column. We will ignore the Address column because it only has text which is not useful for linear regression modeling.

### X and y List
X = HouseDF[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']]

y = HouseDF['Price']

### Split Data into Train, Test
Now we will split our dataset into a training set and testing set using sklearn train_test_split(). the training set will be going to use for training the model and testing set for testing the model. We are creating a split of 40% training data and 60% of the training set.

X_train and y_train contain data for the training model. X_test and y_test contain data for the testing model. X and y are features and target variable names.
 
### Creating and Training the LinearRegression Model
We will import and create sklearn linearmodel LinearRegression object and fit the training dataset in it.
from sklearn.linear_model import LinearRegression 
lm = LinearRegression() 
lm.fit(X_train,y_train) 

OUTPUT
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False) 

### LinearRegression Model Evaluation
Now let’s evaluate the model by checking out its coefficients and how we can interpret them.
print(lm.intercept_)

OUTPUT
-2640159.796851911 

coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient']) coeff_df

What does coefficient of data says:
Holding all other features fixed, a 1 unit increase in Avg. Area Income is associated with an increase of $21.52 .
Holding all other features fixed, a 1 unit increase in Avg. Area House Age is associated with an increase of $164883.28 .
Holding all other features fixed, a 1 unit increase in Avg. Area Number of Rooms is associated with an increase of $122368.67 .
Holding all other features fixed, a 1 unit increase in Avg. Area Number of Bedrooms is associated with an increase of $2233.80 .
Holding all other features fixed, a 1 unit increase in Area Population is associated with an increase of $15.15 .

### Predictions from our Linear Regression Model
Let’s find out the predictions of our test set and see how well it perform.
predictions = lm.predict(X_test)  
plt.scatter(y_test,predictions)

In the above scatter plot, we see data is in a line form, which means our model has done good predictions.
sns.distplot((y_test-predictions),bins=50); 

In the above histogram plot, we see data is in bell shape (Normally Distributed), which means our model has done good predictions.

### Regression Evaluation Metrics
Here are three common evaluation metrics for regression problems:

Mean Absolute Error (MAE) is the mean of the absolute value of the errors:

$$\frac 1n\sum_{i=1}^n|y_i-\hat{y}_i|$$
Mean Squared Error (MSE) is the mean of the squared errors:

$$\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2$$
Root Mean Squared Error (RMSE) is the square root of the mean of the squared errors:

$$\sqrt{\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2}$$
Comparing these metrics:

MAE is the easiest to understand because it’s the average error.
MSE is more popular than MAE because MSE “punishes” larger errors, which tends to be useful in the real world.
RMSE is even more popular than MSE because RMSE is interpretable in the “y” units.
All of these are loss functions because we want to minimize them.

### Conclusion
We have created a Linear Regression Model which we help the real state agent for estimating the house price.
