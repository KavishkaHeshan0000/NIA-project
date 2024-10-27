import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the dataset
file_name = 'Cancer_Data.csv'  # Make sure to upload the file using the Streamlit interface
data = pd.read_csv(file_name)

# Data preprocessing
data = data.drop(columns=['id', 'Unnamed: 32'], errors='ignore')
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Split the data into features and target
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Streamlit app title
st.title('Breast Cancer Data Visualization')

# Sidebar selection for different visualizations
st.sidebar.title('Select Visualization')
visualization_type = st.sidebar.selectbox('Choose the type of visualization', 
                                          ['Distribution of Diagnosis', 
                                           'Correlation Matrix', 
                                           'Pair Plot', 
                                           'Box Plot', 
                                           'Distribution Plot', 
                                           'Violin Plot', 
                                           'Scatter Plot', 
                                           'Feature Importance'])

# Visualization: Distribution of Diagnosis
if visualization_type == 'Distribution of Diagnosis':
    st.subheader('Distribution of Diagnosis')
    plt.figure(figsize=(10, 6))
    sns.countplot(x='diagnosis', data=data, palette=['#FF6347', '#4682B4'])
    plt.title('Distribution of Diagnosis')
    plt.xlabel('Diagnosis (0 = Benign, 1 = Malignant)')
    plt.ylabel('Count')
    st.pyplot(plt)

# Visualization: Correlation Matrix
elif visualization_type == 'Correlation Matrix':
    st.subheader('Correlation Matrix')
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    st.pyplot(plt)

# Visualization: Pair Plot
elif visualization_type == 'Pair Plot':
    st.subheader('Pair Plot of Features')
    sns.pairplot(data, hue='diagnosis', diag_kind='kde', markers=["o", "s"], palette=['#FF6347', '#4682B4'])
    st.pyplot(plt)

# Visualization: Box Plot
elif visualization_type == 'Box Plot':
    st.subheader('Box Plot of Selected Features by Diagnosis')
    
    # Select a feature for the box plot
    selected_feature = st.selectbox("Select a feature to plot", 
                                     ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean'])
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='diagnosis', y=selected_feature, data=data, palette=['#FF6347', '#4682B4'])
    plt.title(f'Box Plot of {selected_feature} by Diagnosis')
    plt.xlabel('Diagnosis (0 = Benign, 1 = Malignant)')
    plt.ylabel(selected_feature)
    st.pyplot(plt)

# Visualization: Distribution Plot
elif visualization_type == 'Distribution Plot':
    st.subheader('Distribution of Selected Feature')
    
    # Select a feature for the distribution plot
    selected_feature = st.selectbox("Select a feature to plot", 
                                     ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean'])
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x=selected_feature, hue='diagnosis', element='step', stat='density', common_norm=False, palette=['#FF6347', '#4682B4'])
    plt.title(f'Distribution of {selected_feature}')
    plt.xlabel(selected_feature)
    plt.ylabel('Density')
    st.pyplot(plt)

# Visualization: Violin Plot
elif visualization_type == 'Violin Plot':
    st.subheader('Violin Plot of Radius Mean by Diagnosis')
    plt.figure(figsize=(15, 8))
    sns.violinplot(x='diagnosis', y='radius_mean', data=data, palette=['#FF6347', '#4682B4'])
    plt.title('Violin Plot of Radius Mean by Diagnosis')
    plt.xlabel('Diagnosis (0 = Benign, 1 = Malignant)')
    plt.ylabel('Radius Mean')
    st.pyplot(plt)

# Visualization: Scatter Plot
elif visualization_type == 'Scatter Plot':
    st.subheader('Scatter Plot of Radius Mean vs. Texture Mean')
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='radius_mean', y='texture_mean', hue='diagnosis', data=data, palette=['#FF6347', '#4682B4'], s=100, alpha=0.6, edgecolor='w')
    plt.title('Scatter Plot of Radius Mean vs. Texture Mean')
    plt.xlabel('Radius Mean')
    plt.ylabel('Texture Mean')
    plt.legend(title='Diagnosis', loc='upper right')
    st.pyplot(plt)

# Visualization: Feature Importance
elif visualization_type == 'Feature Importance':
    # Train the Random Forest model
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    # Get feature importance
    importance = rf_model.feature_importances_

    # Create a DataFrame for visualization
    feature_names = X_train.columns
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    # Visualization
    st.subheader('Feature Importance from Random Forest Model')
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
    plt.title('Feature Importance')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    st.pyplot(plt)

# Display the visualization
st.write("Select a visualization type from the sidebar to view different plots.")