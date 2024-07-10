import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


# read csv file
df = pd.read_csv('water_potability.csv')

print(df.info())

# ph missing data
'contains NaN so using median to fill data'
df['ph'] = df['ph'].fillna(df['ph'].median())

# sulfate missing data
df['Sulfate'] = df['Sulfate'].fillna(df['Sulfate'].median())

# Trihalomethanes missing data
df['Trihalomethanes'] = df['Trihalomethanes'].fillna(df['Trihalomethanes'].median())

# using z-score to remove outliers less tha z-score 3
z_scores = np.abs(stats.zscore(df))
df = df[(z_scores < 3).all(axis = 1)]

print(df.info())

# Selecting Independent (X) and dependent (y) variables:
X = df[['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon',
        'Trihalomethanes', 'Turbidity']]
y = df['Potability']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# checking for best hyper-parameters

# for SVC
SVC_param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'kernel': ['linear', 'rbf', 'poly']
}

SVC_grid_search = GridSearchCV(SVC(), SVC_param_grid, cv= 5, n_jobs=-1)
SVC_grid_search.fit(X_train, y_train)
print("Best parameters for Support Vector Classification is:", SVC_grid_search.best_params_)

# for Logistic Regression

lr_param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs']
}

lr_grid_search = GridSearchCV(LogisticRegression(max_iter = 1000), lr_param_grid, cv= 5, n_jobs=-1)
lr_grid_search.fit(X_train, y_train)

print("Best parameters for Logistic Regression:", lr_grid_search.best_params_)

