# -*- coding: utf-8 -*-
"""NIAFinalResearch.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1HOYz5eml8vRzgJ7qsK7v7ceMtpxCDneC

**Data Loading and Inspection**
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
    
#file upload and connect
data = pd.read_csv('Cancer_Data.csv')
# Inspect the first few rows
print(data.head())
# Get information about data types and missing values

print(data.info())
"""**Data Cleaning**"""

# Drop unnecessary columns
data_cleaned = data.drop(columns=['id', 'Unnamed: 32'], errors='ignore')

# Convert 'diagnosis' column to numeric: 1 for malignant, 0 for benign
data_cleaned['diagnosis'] = data_cleaned['diagnosis'].map({'M': 1, 'B': 0})

# Check for missing values
print(data_cleaned.isnull().sum())

"""**Exploratory Data Analysis (EDA)**

**Distribution of Target Variable**
"""

# Plot the distribution of the target variable
sns.countplot(x='diagnosis', data=data_cleaned, palette='viridis')
plt.title('Distribution of Diagnosis')
plt.xlabel('Diagnosis (0 = Benign, 1 = Malignant)')
plt.ylabel('Count')
plt.show()

"""**Histograms and Density Plots for Numerical Features**"""

# Histograms for all numerical features
data_cleaned.drop(columns=['diagnosis']).hist(bins=30, figsize=(20, 15), color='skyblue')
plt.suptitle('Histograms of Numerical Features')
plt.show()

# Density plot for a specific feature
sns.kdeplot(data=data_cleaned, x='radius_mean', hue='diagnosis', fill=True)
plt.title('Density Plot of Radius Mean by Diagnosis')
plt.show()

"""**Feature Correlation**"""

# Correlation matrix
corr_matrix = data_cleaned.corr()

# Plot the correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Features')
plt.show()

"""**Pair Plot**"""

# Pair plot for a subset of features
sns.pairplot(data_cleaned[['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean']],
             hue='diagnosis', palette='viridis')
plt.suptitle('Pair Plot of Selected Features', y=1.02)
plt.show()

"""**Box Plots for Feature Comparison**"""

# Box plot for a specific feature
plt.figure(figsize=(10, 6))
sns.boxplot(x='diagnosis', y='area_mean', data=data_cleaned, palette='Set3')
plt.title('Box Plot of Area Mean by Diagnosis')
plt.show()

"""**Feature Engineering**

**Normalization/Standardization**
"""

# Standardize the features
features = data_cleaned.drop(columns=['diagnosis'])
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Convert back to DataFrame
features_scaled = pd.DataFrame(features_scaled, columns=features.columns)

"""**Feature Importance Visualization**"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Preprocess the data
# Split the data into features and target
X = data_cleaned.drop(columns=['diagnosis'])
y = data_cleaned['diagnosis']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 2: Train the RandomForest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 3: Plot the feature importance
importances = model.feature_importances_
feature_names = X.columns

# Create a DataFrame for visualization
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Feature Importance')
plt.show()

"""**Splitting the Dataset**"""

# Split the data into features and target
X = features_scaled
y = data_cleaned['diagnosis']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""**Model Selection and Training**"""

# Initialize the model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

"""**Model Evaluation**"""

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Make predictions
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Classification report
print(classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

"""**Hyperparameter Tuning**"""

from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30]
}

# Grid search for best parameters
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
print(f'Best parameters: {grid_search.best_params_}')

import joblib

# Save the trained model
joblib.dump(best_model, 'cancer_prediction_model.pkl')

# To load the model in future
loaded_model = joblib.load('cancer_prediction_model.pkl')
