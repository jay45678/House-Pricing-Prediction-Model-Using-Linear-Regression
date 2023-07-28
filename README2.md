# House-Pricing-Prediction-Model-Using-Linear-Regression

In this you will learn how to create Machine Learning Linear Regression Model. You will be analyzing a house price predication dataset for finding out price of house on different parameters. You will do Exploratory Data Analysis, split the training and testing data, Model Evaluation and Predictions. 

### Problem Statement
A real state agents want the help to predict the house price for regions in the USA. He gave you the dataset to work on and you decided to use Linear Regressioon Model. Create a model which will help him to estimate of what the house would sell for.

Dataset contains 10 columns and 193011 rows with CSV extension. The data contains the following columns :

- 'seller_type': type of seller of the flat.
- 'bedroom': number of bedroom in the same flat.
- 'layout_type': type of layout of contruction of a flat.
- 'property_type': type of apartment in which the flat situated.
- 'locality': Address of the flat where the flat situated in the same city.
- 'price': Price that the flat sold at.
- 'area': Area of the flat in sqft.
- 'furnish_type': Furnishment type of the flat what the flat is, furnished or not.
- 'bathroom': Number of bathroom in the same flat.
- 'city': In which city that flat is situated.

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
sns.pairplot(dataF) : This plot is for all the variable variable between relationship.

sns.distplot(dataF['price']) : This plot is for relationship between count and price.

Plot like heatmap of this data frame to identify the dependent and independent variables from other variables. 
sns.heatmap(dataF.corr(), annot=True) : This plot is for relationship between variables like 'bedroom', 'area', 'bathroom' and 'price'.

### Get Data Ready For Training a Linear Regression Model
Let’s now begin to train out the regression model. We will need to first split up our data into an X list that contains the features to train on, and a y list with the target variable, in this case, the Price column. We will ignore the Address column because it only has text which is not useful for linear regression modeling.

### X and y List
X = dataF[['bedroom', 'area', 'bathroom']]

y = dataF['price']

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
-18370.384640213815



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
