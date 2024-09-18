# -*- coding: utf-8 -*-
"""Cancer_Project

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1gOx3wx5-zPF8fKQNEMfdTcZPDguaPbD_
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#file upload and connect
file_path = '/content/Cancer_Data.csv'
df_cancer = pd.read_csv(Cancer_Data.csv)
df_cancer = pd.read_csv('/path/to/your/Cancer_Data.csv')

#display top table top raws
df_cancer.head()

df_cancer.info() #data type each each data colomns

#count: The number of non-null entries.
#mean: The average of the values.
#std: The standard deviation, which measures the amount of variation.
#min: The minimum value.
#25%: The 25th percentile (first quartile).
#50%: The median (50th percentile, second quartile).
#75%: The 75th percentile (third quartile).
#max: The maximum value.

#This is the statistical summery

df_cancer.describe()

#Searching Null data but this haven't null data.Therefore we not need to fill null colomns before the analizyng the data.
df_cancer.isnull().sum()

#display categorical colomn names
categorical_columns = df_cancer.select_dtypes(include=['object', 'category']).columns.tolist()
print(categorical_columns)

#display Numerical colomn names
numerical_columns = df_cancer.select_dtypes(include=['number']).columns.tolist()
print(numerical_columns)

#Check distribution of target variable
diagnosis_distribution = df_cancer['diagnosis'].value_counts()

# Display the distribution
print(diagnosis_distribution)

# Plot the distribution
plt.figure(figsize=(6,4))
df_cancer['diagnosis'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribution of Diagnosis')
plt.xticks([0, 1], ['Benign', 'Malignant'], rotation=0)
plt.ylabel('Count')
plt.show()

#top table rows
df_cancer.head()

#Drop irrelevant columns
df_cancer.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)

# Convert 'diagnosis' to numerical values (M -> 1, B -> 0)
df_cancer['diagnosis'] = df_cancer['diagnosis'].map({'M': 1, 'B': 0})

#search data types
df_cancer.dtypes

#Discriptive analysis
# Plot histograms for numerical features
df_cancer.hist(bins=20, figsize=(20, 15))
plt.tight_layout()
plt.show()

# Boxplots for a specific feature
plt.figure(figsize=(8, 6))
sns.boxplot(x='diagnosis', y='radius_mean', data=df_cancer)
plt.title('Boxplot of Radius Mean by Diagnosis')
plt.show()

# Pairplot for visual correlation
sns.pairplot(df_cancer, hue='diagnosis', vars=['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean'])
plt.show()

print("Skewness of each feature:\n", df_cancer.skew())
print("\nKurtosis of each feature:\n", df_cancer.kurt())

df_cancer['diagnosis'].value_counts()

plt.figure(figsize=(8, 6))
sns.kdeplot(df_cancer['radius_mean'], shade=True)
plt.title('Density Plot of Radius Mean')
plt.show()

# Boxplot for 'radius_mean'
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_cancer['radius_mean'])
plt.title('Boxplot of Radius Mean')
plt.show()

# Calculate the correlation
correlation_matrix = df_cancer.corr()

# Display the correlation
correlation_matrix

correlation_with_diagnosis = correlation_matrix['diagnosis'].sort_values(ascending=False)

print(correlation_with_diagnosis)

#Plot the heatmap of the correlation
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

#Separate features (X) and target (y)
X = df_cancer.drop('diagnosis', axis=1)
y = df_cancer['diagnosis']

#Feature Scaling Part
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
X_scaled_df.head()

print("Means after scaling:\n", X_scaled_df.mean())
print("\nStandard deviations after scaling:\n", X_scaled_df.std())

#Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#Display the shapes of the train and test sets
print("Training set shape (X_train):", X_train.shape)
print("Training set shape (y_train):", y_train.shape)
print("Test set shape (X_test):", X_test.shape)
print("Test set shape (y_test):", y_test.shape)

#Check distribution in the training set
print("Training set distribution:\n", y_train.value_counts())

#Check distribution in the test set
print("\nTest set distribution:\n", y_test.value_counts())

#Initialize the Logistic Regression model
model = LogisticRegression()

#Train the model on the training data
model.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = model.predict(X_test)

#Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

#Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Generate the classification report
class_report = classification_report(y_test, y_pred)
print("\nClassification Report:\n", class_report)



import pickle
from sklearn.linear_model import LogisticRegression

# Save the model to a file
filename = 'logistic_regression_model.pkl'
pickle.dump(LogisticRegression(), open(filename, 'wb'))
