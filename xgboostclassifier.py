import numpy as np
import shap
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score


# Import Dataset
df = pd.read_csv('attrition_data_clean.csv')
X = df.drop('Attrition', axis=1) 
y = df['Attrition']  

# Split Train and Test Sets
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=123)
for train_index, test_index in splitter.split(X, y):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]

# Separate features and target variables in training and test sets
X_train = strat_train_set.drop('Attrition', axis=1)
y_train = strat_train_set['Attrition']
X_test = strat_test_set.drop('Attrition', axis=1)
y_test = strat_test_set['Attrition']

'''
# Verify the splits
print("Training set:")
print(strat_train_set.head())

print("\nTest set:")
print(strat_test_set.head())
'''

# Train the XGBoost Classifier
clf = xgb.XGBClassifier(random_state=123, eval_metric='mlogloss')
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the classifier
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

# Print results
print("Confusion Matrix:")
print(conf_matrix)
print("\nAccuracy:", accuracy)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)

# Predict probabilities on the entire dataset
y_proba = clf.predict_proba(X)[:, 1]  # Get probabilities for the '1' class

# Set a threshold for high attrition risk
threshold = .95
high_risk_employees = df[y_proba > threshold]

# Export the list of high-risk employees to a CSV file
high_risk_employees.to_csv('high_risk_employees.csv', index=False)

# Compute SHAP values and plot feature importance
explainer = shap.Explainer(clf)
shap_values = explainer(X_train)

# Plot feature importance
shap.summary_plot(shap_values, X_train, plot_type="bar")
plt.show()