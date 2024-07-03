import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, ConfusionMatrixDisplay

# Read .csv file from source
df = pd.read_csv("diabetes.csv")

# Save path for plot figures
save_dir = ('/Users/jayanshrestha/PycharmProjects/jen]/DataScienceFinalProject/FinalProject'
            '/Project1_DiabetesPrediction/Figures/')

# Data Profiling and Inspection
print(df.head())
print(df.info())  # Out of 768 entries 768 are not null. Hence, we conclude that there is no missing data.
print(df.dtypes)
print(df.describe())


# Exploratory Data Analysis and Inspection

# Analyzing Independent Variables (X)
# Defining a function to plot histogram, analyze kurtosis and save figure in a folder

def histogram_plot(feature):
    hist = sns.histplot(data = df, x = feature, hue = 'Outcome', bins = 40, alpha = 0.4,
                        palette = ['#5a5ee6', '#eb6060'], multiple = 'stack', kde = True)
    hist.bar_label(hist.containers[0], fontsize = 7)
    hist.bar_label(hist.containers[1], fontsize = 7)
    kurt = kurtosis(df[feature], fisher = False)
    plt.title(f'kurtosis = {round(kurt, 1)}')
    plt.savefig(os.path.join(save_dir, feature))
    plt.show()


# Executing the function with all the Independent Variables

histogram_plot('Pregnancies')
histogram_plot('Glucose')
histogram_plot('BloodPressure')
histogram_plot('SkinThickness')
histogram_plot('Insulin')
histogram_plot('BMI')
histogram_plot('DiabetesPedigreeFunction')
histogram_plot('Age')


# Analyzing Outcome(y) Dependent Variable

count = sns.countplot(data = df, x = 'Outcome', hue = 'Outcome', palette = ['#8aeb60', '#e65555'],
                      legend = False)  # avoiding deprecation by adding hue and legend= False. Got warning!
count.bar_label(count.containers[0], fontsize = 7)
count.bar_label(count.containers[1], fontsize = 7)
plt.savefig(os.path.join(save_dir, 'Outcome.png'))
print(plt.show())

# Using boxplot to check outliers

plt.figure(figsize = (8, 4))
sns.boxplot(data = df, palette = "Paired", linewidth = 1, fliersize = 5, flierprops = {"marker": "o"})
plt.xticks(rotation = 45, fontsize = 8)
plt.savefig(os.path.join(save_dir, 'Outlier.png'))
print(plt.show())

# Using Pair Plot to visualize the distribution and relations of all the variables.

sns.pairplot(df, hue = 'Outcome', height = 1.5, aspect = 1, palette = ['#8aeb60', '#e65555'])
plt.savefig(os.path.join(save_dir, 'pairplot.png'))
print(plt.show())


# Finding correlation between Outcome and all other variables using heatmap

sns.heatmap(df.corr(numeric_only = True),
            annot = True, cbar_kws = {
        'shrink': 0.8})  # the heatmap shows that there is a moderate correlation between outcome and glucose level.
plt.savefig(os.path.join(save_dir, 'correlation.png'))
print(plt.show())

# Checking for missing values

print(df.isna().sum())  # There is no missing values.

# Selecting dependent and independent variables

X = df.loc[:, ["Glucose", "BMI", "Age"]]  # DataFrame 2-Dimension
y = df['Outcome']  # Series 1-Dimension

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 40)

# Machine Learning Model = Using support Vector Machine(SVM) as this is binary classification

model = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel = 'linear', gamma = 'scale'))])
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# using f1 score to check accuracy and displaying Confusion matrix

print(f'f1_score = {f1_score(y_true = y_test, y_pred = y_pred)}')
print(f'accuracy_score = {accuracy_score(y_true = y_test, y_pred = y_pred)}')

# Evaluating using ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap = "Accent_r")
plt.title("kernel = linear")
plt.savefig(os.path.join(save_dir, 'ConfusionMatrixDisplay_linear.png'))
print(plt.show())