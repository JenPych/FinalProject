import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import kurtosis
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, ConfusionMatrixDisplay

# read csv file
df = pd.read_csv('water_potability.csv')

# save path
save_dir = ("/Users/jayanshrestha/PycharmProjects/jen]/DataScienceFinalProject/FinalProject/Project4_WaterPotability"
            "/Figure_waterpotability")

# data profiling and analysis

pd.set_option('display.max_columns', 20)
print(df.head())
print(df.info())

pd.set_option('display.max_columns', 15)
print(df.describe().transpose())

# check missing data
print(df.isna().sum())

sns.heatmap(df.isna(), cmap = 'Spectral_r')
print(plt.show())

print(df.loc[df.isna().any(axis = 1)])


# Exploratory data analysis

def hist_plot(feature, palette_color):
    sns.histplot(data = df, x = feature, hue = 'Potability', kde = True, palette = palette_color)
    plt.savefig(os.path.join(save_dir, f'{feature}_histplot.png'))
    return plt.show()


def box_plot(feature, palette_color):
    sns.boxplot(data = df, x = 'Potability', y = feature, hue = 'Potability', palette = palette_color,
                flierprops = dict(marker = '+', color = 'red', markersize = 5))

    kurt = kurtosis(df[feature], fisher = False)
    plt.title(f'kurtosis ={round(kurt, 1)}')
    plt.savefig(os.path.join(save_dir, f'{feature}_boxplot.png'))
    return plt.show()


# ph
# ph missing data
'contains NaN so using median to fill data'
df['ph'] = df['ph'].fillna(df['ph'].median())

hist_plot('ph', 'icefire')
box_plot('ph', 'icefire')

# Hardness
hist_plot('Hardness', 'magma')
box_plot('Hardness', 'magma')

# Solids
hist_plot('Solids', 'viridis')
box_plot('Solids', 'viridis')

# Chloramines
hist_plot('Chloramines', 'rocket')
box_plot('Chloramines', 'rocket')

# Sulfate
# sulfate missing data
df['Sulfate'] = df['Sulfate'].fillna(df['Sulfate'].median())

hist_plot('Sulfate', 'cubehelix')
box_plot('Sulfate', 'cubehelix')

# Conductivity
hist_plot('Conductivity', 'crest')
box_plot('Conductivity', 'crest')

# Organic_carbon
hist_plot('Organic_carbon', 'Paired')
box_plot('Organic_carbon', 'Paired')

# Trihalomethanes
# Trihalomethanes missing data
df['Trihalomethanes'] = df['Trihalomethanes'].fillna(df['Trihalomethanes'].median())

hist_plot('Trihalomethanes', 'hls')
box_plot('Trihalomethanes', 'hls')

# Turbidity
hist_plot('Turbidity', 'pastel')
box_plot('Turbidity', 'pastel')

# Potability
count = sns.countplot(data = df, x = 'Potability', hue = 'Potability', palette = 'mako')
count.bar_label(count.containers[0], fontsize = 7)
count.bar_label(count.containers[1], fontsize = 7)
plt.savefig(os.path.join(save_dir, 'potability.png'))
print(plt.show)

# confirming no NaN remains
print(df.isna().sum())

# pairplot
sns.pairplot(data = df, hue = 'Potability', palette = 'Set2')
plt.savefig(os.path.join(save_dir, 'pairplot.png'))
print(plt.show())

# correlation using heatmap
sns.heatmap(df.corr(numeric_only = True), annot = True, annot_kws = {'size': 7}, cmap = 'Spectral_r')
plt.savefig(os.path.join(save_dir, 'correlation.png'))
print(plt.show())

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

# Model Selection: using SVC
model = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel = 'rbf', gamma = 'scale', C = 1.0))])
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f'f1_score = {f1_score(y_true = y_test, y_pred = y_pred)}')
print(f'Accuracy = {accuracy_score(y_true = y_test, y_pred = y_pred)}')

# Confusion Matrix Display

ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap = "flare")
plt.title("kernel = rbf")
plt.savefig(os.path.join(save_dir, 'ConfusionMatrixDisplay_SVC.png'))
print(plt.show())

# Model Selection using Logistic Regression
model = Pipeline([('scaler', StandardScaler()), ('logReg', LogisticRegression())])
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f'f1_score = {f1_score(y_true = y_test, y_pred = y_pred)}')
print(f'Accuracy = {accuracy_score(y_true = y_test, y_pred = y_pred)}')

# Confusion Matrix Display

ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap = "mako")
plt.title("Logistic Regression")
plt.savefig(os.path.join(save_dir, 'ConfusionMatrixDisplay_logReg.png'))
print(plt.show())
