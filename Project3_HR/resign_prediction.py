import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, ConfusionMatrixDisplay

# Loading dataset
df = pd.read_csv('hr.csv')

# Save path for plot figures
save_dir = ('/Users/jayanshrestha/PycharmProjects/jen]/DataScienceFinalProject/FinalProject/Project3_HR/Figures_HR')

# Data Profiling and Analysis
print(df.head())
print(df.info())

# describe all the numeric columns
# pd.set_option('display.max_columns', 100)
print(df.describe().transpose())

# describe all the non-numeric columns
print(df.describe(include = object))

# display all the non-numeric values
print(df['Department'].unique())
print(df['salary'].unique())
print(df['left'].unique())

# Checking for Missing Data
print(df.isna().sum())

# plotting heatmap to show the intensity of missing data
sns.heatmap(df.isna(), cmap = 'coolwarm')
plt.savefig(os.path.join(save_dir, 'NaN_heatmap.png'))
print(plt.show())

# age column missing more than 70% of the data. hence, we drop it
df.drop('age', axis = 1, inplace = True)
print(df.head())  # age column is dropped, useless for ML model

# salary column is also missing some data
print(df.loc[df.isna().any(axis = 1)])  # this shows that salary is missing in 29 rows.

# Visual representation of the 29 missing NaN in Salary
sns.scatterplot(y = df['salary'].isna(), x = df.index, marker = 'x', s = 100, hue = df['salary'].isna(),
                palette = 'mako', legend = False)
plt.ylabel('0 = not NaN, 1 = NaN')
plt.savefig(os.path.join(save_dir, 'NaN_salary_column.png'))
print(plt.show())

#  since only 29 rows out of 15004 rows are empty, we can drop NaN salary rows
df.dropna(subset = ['salary'], inplace = True)
print(df.isna().sum())  # all 29 rows has been removed.

# ? in left column.
print(df.loc[df['left'] == '?'].transpose())
'''there are 4 rows with '?' values in out target variable.
these are unknown cases in our target variable. it is sensible to consider this as a NaN value and we should
drop such rows '''

# dropping entire rows with target = ? as this is our dependent variable.
df.drop(df[df['left'] == '?'].index, inplace = True)
print(f'Any = {df.loc[df['left'] == '?']}')

# Encoding Ordinal Data to Numerical.
'''This is a necessary step to convert categorical data into quantitative data
 for our ML model to understand and give us better output'''
# salary and left is object which means they need to be converted into int.

label_encoder = LabelEncoder()
df['salary_encoded'] = label_encoder.fit_transform(df['salary'])
custom_mapping = {'low': 0, 'medium': 1, 'high': 2}
df['salary_encoded'] = df['salary'].map(custom_mapping)

print(df[['salary', 'salary_encoded']].head())  # comparing a newly created column with its origin

df['left_encoded'] = label_encoder.fit_transform(df['left'])
custom_mapping2 = {'0': 0, '1': 1}
df['left_encoded'] = df['left'].map(custom_mapping2)

print(df[['left', 'left_encoded']].head())  # comparing a newly created column with its origin


# Exploratory Data Analysis and Inspection

# creating a function for count_plot

def count_plot(feature, color_palette):
    count_p = sns.countplot(data = df, x = feature, hue = 'left', palette = color_palette)
    count_p.bar_label(count_p.containers[0], fontsize = 7)
    count_p.bar_label(count_p.containers[1], fontsize = 7)
    plt.xticks(rotation = 45, fontsize = 7)
    plt.savefig(os.path.join(save_dir, feature))
    print(plt.show())


# creating function for hist_plot

def hist_plot(feature, color_palette):
    count_h = sns.histplot(data = df, x = df[feature], hue = 'left', palette = color_palette)
    count_h.bar_label(count_h.containers[0], fontsize = 7, rotation = 90)
    count_h.bar_label(count_h.containers[1], fontsize = 7, rotation = 90)
    plt.savefig(os.path.join(save_dir, (f'hist_{feature}.png')))
    print(plt.show())


# creating function for boxplot

def box_plot(feature, color_palette):
    sns.boxplot(data = df, x = df['left'], y = df[feature], hue = 'left', palette = color_palette)
    plt.title(f'average = {df[feature].mean()}')
    plt.savefig(os.path.join(save_dir, (f'box_{feature}.png')))
    print(plt.show())


# Analysing Independent Variables (X)

# satisfaction level
hist_plot('satisfaction_level', 'icefire')
box_plot('satisfaction_level', 'icefire')

# last evaluation
hist_plot('last_evaluation', 'vlag')
box_plot('last_evaluation', 'vlag')

# hours worked in a month
hist_plot('average_montly_hours', 'Spectral')
box_plot('average_montly_hours', 'Spectral')

# number of projects
count_plot('number_project', 'magma')


# time spent in the company
count_plot('time_spend_company', 'hls')
box_plot('time_spend_company', 'hls')

# work accident
count_plot('Work_accident', 'Set2')


# promotion in the last 5 years
count_plot('promotion_last_5years', 'rocket')


# Department
count_plot('Department', 'cubehelix')


# Salary
count_plot('salary', 'viridis')


# salary_encoded
count_plot('salary_encoded', 'viridis')


# Analyzing Dependent Variable (Y)

count = sns.countplot(data = df, x = 'left_encoded', hue = 'left', palette = 'pastel')
count.bar_label(count.containers[0], fontsize = 7)
count.bar_label(count.containers[1], fontsize = 7)
plt.savefig(os.path.join(save_dir, 'left_encoded.png'))
print(plt.show())

# Finding correlation
sns.heatmap(df.corr(numeric_only = True), annot = True, cmap = 'Spectral')
plt.savefig(os.path.join(save_dir, 'correlation.png'))
print(plt.show())

# Selecting features (X) and target (y)

X = df[
    ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company',
     'Work_accident', 'promotion_last_5years', 'salary_encoded']]
y = df['left_encoded']

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#  Machine Learning Model
# using SVC

model = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel = 'rbf', gamma = 'scale'))])
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#  Evaluation and Conclusion

# using f1 score to check accuracy and displaying Confusion matrix

print(f'f1_score = {f1_score(y_true = y_test, y_pred = y_pred)}')
print(f'accuracy_score = {accuracy_score(y_true = y_test, y_pred = y_pred)}')

# Evaluating using ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap = "Accent_r")
plt.title("kernel = rbf")
plt.savefig(os.path.join(save_dir, 'ConfusionMatrixDisplay_rbf.png'))
print(plt.show())


# Selecting features (X) and target (y)

X = df[
    ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company',
     'Work_accident', 'promotion_last_5years', 'salary_encoded']]
y = df['left_encoded']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#  Machine Learning Model
# using Logistic Regression

model = Pipeline([('scaler', StandardScaler()), ('LogReg', LogisticRegression())])
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#  Evaluation and Conclusion

# using f1 score to check accuracy and displaying Confusion matrix

print(f'f1_score = {f1_score(y_true = y_test, y_pred = y_pred)}')
print(f'accuracy_score = {accuracy_score(y_true = y_test, y_pred = y_pred)}')

# Evaluating using ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap = "CMRmap")
plt.title("Logistic Regression")
plt.savefig(os.path.join(save_dir, 'ConfusionMatrixDisplay_LogReg.png'))
print(plt.show())

