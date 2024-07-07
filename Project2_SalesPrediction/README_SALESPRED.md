# FinalProject_2

# Sales Prediction Study:

In this study, the objective of the dataset is the cost of advertising in 3 different platforms such as
television, newspaper and radio. The data also provides the Sales of a product as a result of advertising in these platform.
Our objective is to create a Machine Learning Model based on the given data to predict the sales when we advertise the product in these platform.
We will also create a streamlit app for users convenience.


# Libraries used for this study

import os \
import pandas as pd \
import numpy as np
import matplotlib.pyplot as plt\
import seaborn as sns\
from scipy import stats
from sklearn.model_selection import train_test_split\
from sklearn.liner_model import LinearRegression
from sklearn.svm import SVR\
from sklearn.pipeline import Pipeline\
from sklearn.preprocessing import StandardScaler\
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2score
import pickle

# Variables in the data

The datasets consist of one target(dependent) variable that is Sales.
The other independent variables(features) are:
* TV
* Radio
* Newspaper

# Data Inspection and Analysis

Steps to follow:

1. Indepth study of all the independent variables (feature)
2. Uni-variable study of dependent variable ('sales') and figure out what type of variable it is.
3. Multi-variable study to find the relation between dependent (y) and independent (X) variable.
4. Cleaning the data if there is missing data (NaN), addressing outliers 
5. Selecting Independent and Dependent Variables
6. Splitting data into Training and testing sets
7. Algorithm Selection and Model Fitting
8. Check Accuracy of the model
9. Evaluate the results and Conclusion

# Step 1: Study of Independent Variables(features):

* TV: Continuous data

![TV.png](Project2_SalesPrediction/Figure_Sales/TV.png)

The figure above shows the relation between cost of advert in Television and Sales output from it.
Upon observing, there is a positive linear increment in sales as the cost of advert increases.
We can expect high correlation with sales and a strong candidate for the main feature for ML model.

* Radio: Continuous data

![Radio.png](Project2_SalesPrediction/Figure_Sales/Radio.png)

The figure above shows the relation between cost of advert in Radio and Sales output from it.
Upon observing, there seems like a positive increment in sales as the cost of advert increases.
However, it is not an upward trend and can be considered as a moderate feature for sales prediction.
We can expect somewhat moderate to weak correlation with sales and a potential candidate for secondary feature for ML model.

* Newspaper: Continuous data

![Newspaper.png](Project2_SalesPrediction/Figure_Sales/Newspaper.png)

The figure above shows the relation between cost of advert in newspaper and Sales output from it.
Upon observing, the data is spread out and linear relation cannot be drawn from it. 
There seems to have lack of linearity and there is a possibility of ignoring this as a feature for sales prediction.
We can expect somewhat weak or no correlation with sales and a potential skip for ML model.

# Step 2: Studying the dependent variable (y)

* Target = Sales: Continuous data

![sales.png](Project2_SalesPrediction/Figure_Sales/sales.png)

The figure above shows sales count with a minimum of 1.60 and maximum sales of 27.00.
Upon observation, the data seems to be normally distributed as most of the data lies around the middle.

As the data is continuous, the following Machine Learning algorithms will be implemented:

1. Logistic Regression
2. Support Vector Regression

# Step 3: Finding the Relation between feature (X) and target (y):

Based on initial observation, Television should have a high correlation with sales and also radio
to some extent. However, looking into the scatterplot of newspaper, I could not find any distinction 
in distribution. Therefore, it may be possible, we can take this out of consideration.

Let's plot some diagrams to visualize and confirm our assumptions.

* pairplot

![pairplot.png](Project2_SalesPrediction/Figure_Sales/pairplot.png)

Here in the pairplot, we can confirm that TV and radio has positive correlation with sales and
newspaper does not seem to have or has a weak correlation. Also, we can see that TV and radio is 
evenly distributed for which the same cannot be said for newspaper.

Let's check if there is any outliers in the data.

* boxplot

![boxplot.png](Project2_SalesPrediction/Figure_Sales/boxplot.png)

The boxplot here shows that there are outliers in the newspaper data further diminishing its
case to be one of the features. Also, mean is higher in TV compared to others, indicating that
its stance to be one of the main feature is strong. Having a lower mean doesnt indicate that radio 
will be out-casted as well. We will make further cases to check if its correct to include radio or not.

* heatmap to show correlation of numeric data

![correlation.png](Project2_SalesPrediction/Figure_Sales/correlation.png)

Now, this makes it clearer. We can see that there is high correlation coefficient of 
0.78 between TV and sales and also no or very weak correlation with other independent variable.
Therefore, we will definitely have TV as our main feature in or ML model.

Let's look into radio. It also has a positive correlation with sales with 0.58 correlation coefficient.
Although it isnt as high as TV, it isnt close to 0 either. Since, it also doent have outliers, we will 
adjust radio as our second feature and evaluate what is the best.

As for newspaper, correlation coefficient of 0.23 is relatively weak compared to others and also has outliers.
For a final chance, let's try to adjust its outliers using z-score and check if its correlation improves.

Handling outliers using z-score 

from scipy import stats \
z_score = np.abs(stats.zscore(df['newspaper']))\
outlier = df[z_score > 3]\
median_wout_outlier = df['newspaper'][z_score <= 3].median() \
df.loc[z_score > 3, 'newspaper'] = median_wout_outlier

Now, lets visualize the boxplot

![npaper_noutlier_check.png](Project2_SalesPrediction/Figure_Sales/npaper_noutlier_check.png)

The figure above shows that we have addressed the outliers by replacing
z-score greater than 3 with median od the data.

Now, lets see if the correlation has improved significantly. If it does, 
newspaper will be considered as feature for our model.

![corr_after_outlier.png](Project2_SalesPrediction/Figure_Sales/corr_after_outlier.png)

As you can see in the figure, correlation has dropped slightly by 0.1.
Now its confirmed, there is no point in taking newspaper as a feature anymore.

We will go ahead with TV as our main feature with radio alongside it.

# Step 4: Check for missing data

print(df.isna().sum())

There is no missing values in this data. Therefore, no further action is required.


# 5. Selecting Independent and Dependent Variables

Feature Engineering assuming TV/Radio interaction as impact on sales

Creating Interaction feature:

df['TV_radio_interaction'] = df['TV'] * df['radio']

We did this considering in real life scenario, choosing to advertise in multiple
platform will definitely lead to more reach resulting in more sales. As newpaper 
has already been out-casted as a feature, we created an interaction feature as a bonus 
feature for out ML model.

X = df.loc[:, ["TV_radio_interaction", "TV", "radio"]]  # DataFrame 2-Dimension \

y = df['sales']  # Series 1-Dimension

Selecting TV, radio and interaction between TV and radio as 3 features (X)
and target variable (y) as sales.

# 6. Splitting data into Training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# 7. Algorithm Selection and Model Fitting

* Machine Learning Model: Using Linear Regression

rom sklearn.pipeline import Pipeline \
from sklearn.preprocessing import StandardScaler \
from sklearn.linear_model import LinearRegression 

model = Pipeline([('scaler', StandardScaler()), ('linreg', LinearRegression())])\
model.fit(X_train, y_train)\ 

y_pred = model.predict(X_test)\
compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})\
print(compare)

* Machine Learning model: Using Support Vector Regression(SVR)

model = Pipeline([('scaler', StandardScaler()), ('svm', SVR(kernel = 'rbf', gamma = 'scale'))]) \
model.fit(X_train, y_train)

y_pred = model.predict(X_test)\
compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})\
print(compare)

# 8. Check Accuracy of the model: using MSE, MAE and r2 score

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

print(f'mean_squared_error = {mean_squared_error(y_true = y_test, y_pred = y_pred)}') \
print(f'mean_absolute_error = {mean_absolute_error(y_true = y_test, y_pred = y_pred)}') \
print(f'r2_score = {r2_score(y_true = y_test, y_pred = y_pred)}')

* Results of Linear Regression:

mean_squared_error = 0.8127 \
mean_absolute_error = 0.6663 \
r2_score = 0.9742

* Results of Support Vector Regression:

mean_squared_error = 0.9447 \
mean_absolute_error = 0.6729 \
r2_score = 0.9700 

# 9. Evaluate the results and Conclusion

Looking into the above results, it seems like both Linear Regression and Support Vector Regression
has amazing accuracy. However, LR has and edge over SVR in terms of r2_score by 0.42. 

SVR has better MSE and MAE overall.

I also need to address that using 'rbf' kernel gave better results than 'linear' kernel while using SVR.
It is also important to mention that, multiple features were tested. However, the combination of
TV, radio and interaction between TV and radio as 3 features yielded the best result.


# 10. Creating a streamlit app

Model dumping

import pickle \
with open("modelLR.pickle", 'wb') as file: \
    pickle.dump(model, file)

This creates a '.pickle' file type with Linear Regression model. 

import pickle \
with open("modelLR.pickle", 'wb') as file: \
    pickle.dump(model, file)

This creates a '.pickle' file type with Support Vector Regression model. 

We then create a python file in the same directory where '.pickle' file exists adn write the following code:

import streamlit as st \
import pickle \

ask = st.text_input("Which model would you like to use? LR for Linear Regression or SVR for Support Vector Regression") \


if ask == "LR": \
    with open('modelLR.pickle', 'rb') as file: \
        model = pickle.load(file) \

elif ask == "SVR": \
    with open('modelSVR.pickle', 'rb') as file: \
        model = pickle.load(file) \
else: \
    print("error") \

st.write("Ads in TV and Radio") \
tv = st.number_input("Enter the cost of ads on TV: ") \
radio = st.number_input("Enter the cost of ads on radio: ") \
both = tv * radio \

if st.button("Predict"): \
    y_pred = model.predict([[tv, radio, both]]) \
    st.write(f'Predicted Sales = {y_pred}') \

Here we have made a streamlit app asking the user to input which model they would like to use.
Then, the app takes 2 variables in the form of TV ads cost and radio advert cost and finally predicts 
the sales based on the ML model the user selects.

To run the app type the following in the Terminal:

streamlit run pythonfilename.py










