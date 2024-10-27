import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_name = 'Cancer_Data.csv'  # Make sure to upload the file using the Streamlit interface
data = pd.read_csv(file_name)

# Data preprocessing
data = data.drop(columns=['id', 'Unnamed: 32'], errors='ignore')
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

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
                                           'Scatter Plot'])

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
    
    # Select features for pair plot
    features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']
    selected_feature_x = st.selectbox("Select feature for X-axis", features)
    selected_feature_y = st.selectbox("Select feature for Y-axis", features)

    # Ensure that X and Y features are not the same
    while selected_feature_x == selected_feature_y:
        st.warning("Please select different features for X and Y axes.")
        selected_feature_y = st.selectbox("Select feature for Y-axis", features)
    
    # Create the scatter plot for selected features
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data[selected_feature_x], y=data[selected_feature_y], hue=data['diagnosis'], palette=['#FF6347', '#4682B4'], s=100, alpha=0.6, edgecolor='w')
    plt.title(f'Scatter Plot of {selected_feature_x} vs. {selected_feature_y}')
    plt.xlabel(selected_feature_x)
    plt.ylabel(selected_feature_y)
    plt.legend(title='Diagnosis', loc='upper right')
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
    st.subheader('Distribution of Features')
    selected_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']
    plt.figure(figsize=(12, 8))
    for i, feature in enumerate(selected_features):
        plt.subplot(2, 3, i + 1)
        sns.histplot(data=data, x=feature, hue='diagnosis', element='step', stat='density', common_norm=False, palette=['#FF6347', '#4682B4'])
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Density')
    plt.tight_layout()
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

# Display the visualization
st.write("Select a visualization type from the sidebar to view different plots.")
