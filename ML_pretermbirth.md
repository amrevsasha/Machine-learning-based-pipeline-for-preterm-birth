import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt

# load the dataset
data = pd.read_csv(r'C:\Users\91800\Dropbox\PC\Documents\Book2.csv')


data['Outcome'] = data['Outcome'].map({'Preterm Birth': 0, 'Term Birth': 1})
bins = (0, 2, 8)
group_names = ['bad', 'good']
data['Outcome'] = pd.cut(data['Outcome'], bins=bins, labels=group_names)
data['Outcome'].unique()
label_quality=LabelEncoder()
data['Outcome']= label_quality.fit_transform(data['Outcome'])

data = data.dropna()

data['Age'] = pd.to_numeric(data['Age'], errors='coerce')
median_age = data['Age'].median()
data['Age'] = data['Age'].fillna(median_age).astype(float)

# encode categorical variables
cat_cols = data.select_dtypes(include='object').columns.tolist()
le = LabelEncoder()
for col in cat_cols:
    data[col] = le.fit_transform(data[col])

# split the data into train and test sets
X = data.drop('Outcome', axis=1)
y = data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# define models to evaluate
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

# evaluate models and store results
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    
    results[name] = {'Accuracy': acc, 'F1 Score': f1, 'Recall': recall, 'AUC': auc, 'FPR': fpr, 'TPR': tpr, 'Thresholds': thresholds}

# plot combined ROC curve and AUC
plt.figure(figsize=(8,8))
for name, result in results.items():
    plt.plot(result['FPR'], result['TPR'], label='{} (AUC = {:.2f})'.format(name, result['AUC']))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {}
    results[name]['Accuracy'] = accuracy_score(y_test, y_pred)
    results[name]['Recall'] = recall_score(y_test, y_pred)
    results[name]['F1 Score'] = f1_score(y_test, y_pred)
    results[name]['AUC'] = roc_auc_score(y_test, y_pred)
    
    print(name)
    for metric, value in results[name].items():
        if metric == 'AUC':
            if isinstance(value, np.ndarray) and len(value) > 1:
                value = np.mean(value)
            else:
                value = value.item()
        value = round(float(value), 2)
        print('{}: {}'.format(metric, value))
    print()
import seaborn as sns

# create a correlation matrix
corr = data.corr()

# create heatmap
plt.figure(figsize=(100, 100))
sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f', annot_kws={'size': 10})

# adjust font size of x and y axis labels
plt.xticks(fontsize=70)
plt.yticks(fontsize=70)

# add spacing between the feature names
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8, wspace=0.3, hspace=0.3)

plt.show()



# Compute ANOVA F-value and p-value between each feature and the target variable
f_values, p_values = f_classif(X, y)

# Choose a significance level (e.g. 0.05)
alpha = 0.1

# Find the significant features
sig_features = []
for i, pval in enumerate(p_values):
    if pval < alpha:
        sig_features.append(X.columns[i])

print(f"Significant features: {sig_features}")


# Initialize and fit the Random Forest Classifier
rf = RandomForestClassifier()
rf.fit(X, y)

# Get feature importances
importances = rf.feature_importances_
feature_names = X.columns

# Sort feature importances in descending order
sorted_indices = np.argsort(importances)[::-1]
sorted_importances = importances[sorted_indices]
sorted_feature_names = feature_names[sorted_indices]

# Define a colormap
colormap = plt.cm.get_cmap('coolwarm')

# Plot the graph with vertical bars of varying colors
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(range(len(sorted_importances)), sorted_importances)

# Set the colors of the bars based on the colormap
for i, bar in enumerate(bars):
    bar.set_color(colormap(sorted_importances[i]))

plt.xticks(range(len(sorted_importances)), sorted_feature_names, rotation='vertical')
plt.ylabel('Importance')
plt.xlabel('Features')
plt.title('Feature Importance - Random Forest Classifier')
plt.grid(True)
plt.show()


# input new data
new_data = pd.DataFrame(columns=X.columns)
for col in new_data.columns:
    value = input(f'Enter value for {col}: ')
    new_data[col] = [value]

# preprocess new data
for col in cat_cols:
    new_data[col] = le.transform(new_data[col])

# make predictions using all models
predictions = {}
for name, model in models.items():
    model.fit(X, y)
    pred = model.predict(new_data)
    predictions[name] = pred

# print predictions
print('Predictions:')
for name, pred in predictions.items():
    print('{}: {}'.format(name, pred))
