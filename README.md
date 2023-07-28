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
