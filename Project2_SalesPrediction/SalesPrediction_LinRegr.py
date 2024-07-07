import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle

# read .csv file
df = pd.read_csv("ads.csv")

# Save path for plot figures
save_dir = ('/Users/jayanshrestha/PycharmProjects/jen]/DataScienceFinalProject/FinalProject'
            '/Project2_SalesPrediction/Figure_Sales/')

# Data profiling and inspection
print(df.head())
print(df.info())
print(df.dtypes)

# dropping and unnamed column as it is only used for indexing

df = df.drop(columns = ['Unnamed: 0'], axis = 1)
print(df.head())

print(df.describe(()))

# Exploratory Data Analysis and Inspection

# Analyzing Independent Variables (X)

# TV advert Sales
sns.scatterplot(data = df, x = df['TV'], y = df['sales'], palette = 'viridis', hue = df['sales'])
plt.title('TV Sales Trend')
plt.savefig(os.path.join(save_dir, 'TV.png'))
print(plt.show())

# Radio advert Sales
sns.scatterplot(data = df, x = df['radio'], y = df['sales'], palette = 'rocket', hue = df['sales'])
plt.title('Radio Sales Trend')
plt.savefig(os.path.join(save_dir, 'Radio.png'))
print(plt.show())

# Newspaper advert sales
sns.scatterplot(data = df, x = df['newspaper'], y = df['sales'], palette = 'crest', hue = df['sales'])
plt.title('Newspaper Sales Trend')
plt.savefig(os.path.join(save_dir, 'Newspaper.png'))
print(plt.show())

# Analyzing Dependent Variables (y)
sns.histplot(df['sales'], bins = 20, color = "#72f772", edgecolor = 'black', kde = True)
plt.title("Sales Count")
plt.savefig(os.path.join(save_dir, 'sales.png'))
print(plt.show())

# Checking for missing values
print(df.isna().sum())  # There is no missing values.

# boxplot for checking outliers
sns.boxplot(data = df, palette = 'magma', linewidth = 1, fliersize = 5, flierprops = {"marker": "x"})
plt.savefig(os.path.join(save_dir, 'boxplot.png'))
print(plt.show())

# pair-plot to find relation between all variables
sns.pairplot(data = df, plot_kws = {'color': "#72f772"}, diag_kws = {'color': "#9578bf"})
plt.savefig(os.path.join(save_dir, 'pairplot.png'))
print(plt.show())

# heatmap to check correlation
sns.heatmap(df.corr(numeric_only = True), annot = True, cmap = 'mako')
plt.savefig(os.path.join(save_dir, 'correlation.png'))
print(plt.show())


# Handling outliers using z-score
z_score = np.abs(stats.zscore(df['newspaper']))
outlier = df[z_score > 3]
median_wout_outlier = df['newspaper'][z_score <= 3].median()
df.loc[z_score > 3, 'newspaper'] = median_wout_outlier

sns.boxplot(data = df['newspaper'], flierprops = {"marker": "x"})
plt.title('Newspaper Outlier Check')
plt.savefig(os.path.join(save_dir, 'npaper_noutlier_check.png'))
print(plt.show())

# heatmap after addressing outliers
sns.heatmap(df.corr(numeric_only = True), annot = True)
plt.savefig(os.path.join(save_dir, 'corr_after_outlier.png'))
print(plt.show())

# Feature Engineering assuming TV/Radio interaction as impact on sales
# Creating Interaction feature
df['TV_radio_interaction'] = df['TV'] * df['radio']

# Selecting dependent and independent variables
X = df.loc[:, ["TV_radio_interaction", "TV", "radio"]]  # DataFrame 2-Dimension
y = df['sales']  # Series 1-Dimension

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Machine Learning Model: Using Linear Regression
model = Pipeline([('scaler', StandardScaler()), ('linreg', LinearRegression())])
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(compare)

# using MSE, MAE and r2 score to check accuracy
print("Using Linear Regression")
print(f'mean_squared_error = {mean_squared_error(y_true = y_test, y_pred = y_pred)}')
print(f'mean_absolute_error = {mean_absolute_error(y_true = y_test, y_pred = y_pred)}')
print(f'r2_score = {r2_score(y_true = y_test, y_pred = y_pred)}')

# Model dumping
with open("modelLR.pickle", 'wb') as file:
    pickle.dump(model, file)
