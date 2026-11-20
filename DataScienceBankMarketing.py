"""
Bank Marketing - Data Science Project (UCI)

Binary classification: predicting term deposit subscription
Methods: Logistic Regression and Decision Tree

@author: Russel Issembe
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
import io
import urllib.request
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, RocCurveDisplay


####                           #####
#    Part 1: The dataset         #
####                           #####

### Importing the data ###
# The bank.csv file is saved in the same folder as this script after downloading.
folder = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
file = folder / 'bank.csv'

if not file.exists():
    url = 'https://archive.ics.uci.edu/static/public/222/bank+marketing.zip'
    with urllib.request.urlopen(url, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
        with zipfile.ZipFile(io.BytesIO(archive.read('bank.zip'))) as bank_archive:
            file.write_bytes(bank_archive.read('bank.csv'))

df = pd.read_csv(file, sep=';')

print('Dataset dimensions:', df.shape)
print(df.head())
df.info()
print('\nMissing values:\n', df.isnull().sum())
print('\nDescriptive statistics:\n', df.describe())

# 'unknown' is a category in this dataset, not a missing NaN value.


####                                #####
#    Part 2: Descriptive analysis    #
####                                #####

### Distribution of the target variable ###
print('\nTarget distribution:\n', df['y'].value_counts())
print('\nPercentages:\n', (df['y'].value_counts(normalize=True)*100).round(2))

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x='y', order=['no', 'yes'])
plt.title('Term Deposit Subscription')
plt.xlabel('Subscription (No / Yes)')
plt.ylabel('Number of Customers')
plt.tight_layout()
plt.show()

### Age distribution ###
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='age', bins=25)
plt.title('Age Distribution of Customers')
plt.xlabel('Age')
plt.ylabel('Number of Customers')
plt.tight_layout()
plt.show()

### Comparing subscription rates by job ###
job_rate = df.groupby('job')['y'].apply(lambda x: (x == 'yes').mean()*100)
job_rate = job_rate.sort_values(ascending=False)
print('\nSubscription rate by job:\n', job_rate.round(2))

plt.figure(figsize=(9, 5))
sns.barplot(x=job_rate.values, y=job_rate.index)
plt.title('Subscription Rate by Job')
plt.xlabel('Subscription Rate (%)')
plt.ylabel('Job')
plt.tight_layout()
plt.show()

### Comparing subscription rates by education level ###
education_rate = df.groupby('education')['y'].apply(lambda x: (x == 'yes').mean()*100)
print('\nSubscription rate by education:\n', education_rate.round(2))

plt.figure(figsize=(8, 5))
sns.barplot(x=education_rate.index, y=education_rate.values)
plt.title('Subscription Rate by Education')
plt.xlabel('Education')
plt.ylabel('Subscription Rate (%)')
plt.tight_layout()
plt.show()

### Correlation between numerical variables ###
# 'duration' is shown here for description but excluded from prediction.
plt.figure(figsize=(9, 6))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()


####                                #####
#    Part 3: Data preparation      #
####                                #####

# y = 1: the customer subscribes; y = 0: the customer does not subscribe.
y = df['y'].map({'no': 0, 'yes': 1})

# Remove duration: call length is not available before the call takes place.
X = df.drop(columns=['y', 'duration'])

# Use 80% of the data for training and 20% for testing.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Separate numerical and categorical variables.
numeric = X.select_dtypes(include='number').columns.tolist()
categorical = X.select_dtypes(exclude='number').columns.tolist()

# Standardize numerical variables and encode categorical variables.
preprocessing = ColumnTransformer([
    ('numeric', StandardScaler(), numeric),
    ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical)
])

print('\nTraining observations:', len(X_train))
print('Testing observations:', len(X_test))


####                                   #####
#    Part 4: Model estimation        #
####                                   #####

### Model 1: Logistic Regression ###
logistic = Pipeline([
    ('preprocessing', preprocessing),
    ('model', LogisticRegression(max_iter=2000, random_state=42))
])
logistic.fit(X_train, y_train)
logistic_pred = logistic.predict(X_test)
logistic_prob = logistic.predict_proba(X_test)[:, 1]

print('\nLogistic Regression:')
print(classification_report(y_test, logistic_pred, target_names=['No', 'Yes'], zero_division=0))

### Model 2: Decision Tree ###
tree = Pipeline([
    ('preprocessing', preprocessing),
    ('model', DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=42))
])
tree.fit(X_train, y_train)
tree_pred = tree.predict(X_test)
tree_prob = tree.predict_proba(X_test)[:, 1]

print('\nDecision Tree:')
print(classification_report(y_test, tree_pred, target_names=['No', 'Yes'], zero_division=0))


####                                    #####
#    Part 5: Comparing results         #
####                                    #####

results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree'],
    'Accuracy': [accuracy_score(y_test, logistic_pred), accuracy_score(y_test, tree_pred)],
    'Precision': [precision_score(y_test, logistic_pred, zero_division=0), precision_score(y_test, tree_pred, zero_division=0)],
    'Recall': [recall_score(y_test, logistic_pred, zero_division=0), recall_score(y_test, tree_pred, zero_division=0)],
    'F1-score': [f1_score(y_test, logistic_pred, zero_division=0), f1_score(y_test, tree_pred, zero_division=0)],
    'ROC-AUC': [roc_auc_score(y_test, logistic_prob), roc_auc_score(y_test, tree_prob)]
})
print('\nModel comparison:\n', results.round(3).to_string(index=False))

results.set_index('Model').T.plot(kind='bar', figsize=(10, 5), rot=0)
plt.title('Comparison of Classification Models')
plt.xlabel('Evaluation Metrics')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

### Confusion matrices ###
for name, predictions in [('Logistic Regression', logistic_pred), ('Decision Tree', tree_pred)]:
    cm = confusion_matrix(y_test, predictions)
    ConfusionMatrixDisplay(cm, display_labels=['No', 'Yes']).plot(cmap='Blues', colorbar=False)
    plt.title('Confusion Matrix - ' + name)
    plt.tight_layout()
    plt.show()

### ROC curves ###
fig, ax = plt.subplots(figsize=(7, 5))
RocCurveDisplay.from_predictions(y_test, logistic_prob, name='Logistic Regression', ax=ax)
RocCurveDisplay.from_predictions(y_test, tree_prob, name='Decision Tree', ax=ax)
plt.title('ROC Curves')
plt.tight_layout()
plt.show()

### Displaying the first levels of the decision tree ###
features = tree.named_steps['preprocessing'].get_feature_names_out()
plt.figure(figsize=(17, 8))
plot_tree(tree.named_steps['model'], feature_names=features,
          class_names=['No', 'Yes'], filled=True, max_depth=2, fontsize=8)
plt.title('Decision Tree - First Levels')
plt.tight_layout()
plt.show()

# Interpretation: compare the F1-score and recall for the 'Yes' class.
# The classes are imbalanced, so accuracy alone is not enough.
# The results are based on a single train/test split.
