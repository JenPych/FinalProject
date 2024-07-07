import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle


# Load dataset
df = pd.read_csv('ads.csv')

print(df.head())

# Feature Engineering
# Creating Interaction feature
df['TV_radio_interaction'] = df['TV'] * df['radio']

print(df.describe())
# Selecting dependent and independent variables
X = df.loc[:, ["TV", "radio", "TV_radio_interaction"]]  # DataFrame 2-Dimension
y = df['sales']  # Series 1-Dimension

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Machine Learning model: Using Support Vector Regression(SVR)
model = Pipeline([('scaler', StandardScaler()), ('svm', SVR(kernel = 'rbf', gamma = 'scale'))])
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# using MSE, MAE and r2 score to check accuracy
print("Using Support Vector Regression")
print(f'mean_squared_error = {mean_squared_error(y_true = y_test, y_pred = y_pred)}')
print(f'mean_absolute_error = {mean_absolute_error(y_true = y_test, y_pred = y_pred)}')
print(f'r2_score = {r2_score(y_true = y_test, y_pred = y_pred)}')

# Model dumping
with open("modelSVR.pickle", 'wb') as file:
    pickle.dump(model, file)
