import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression

st.title("Machine Learning ML Application")
st.markdown("""
### About This Application
This interactive application demonstrates the different elements of Logistic Regression.
You can:
- Use one of Seaborn's pre-loaded datasets like the Iris, Titanic, Penguins, or Taxis dataset, or upload your own csv.file!
- Input different feature and target variables to explore the elements of Logistic Regression models.
- Discover binary classification results after selecting categorical and continuous variables for feature and target variables.
""")

### Download or Upload DataSet ###

def load_and_preprocess_data():
    file = st.radio("Choose a pre-loaded dataset from Seaborn or upload your own csv.file", options = ['Seaborn dataset', 'Upload csv.file'])
    # Option 1: Insert your own dataset
    if file == 'Seaborn dataset':
        dataset_names = ['iris', 'titanic', 'penguins', 'taxis'] # curate what Seaborn datasets I want to allow users to choose from
        Seaborn_dataset = st.selectbox("Choose a Seaborn dataset:", dataset_names) # streamlit widget
        if Seaborn_dataset:
            df = sns.load_dataset(Seaborn_dataset)
    else:
        user_file = st.file_uploader('Upload a csv file', type = 'csv') # use streamlit widget for uploading files - set to only accepting csv
        if user_file: # if the user uploads a file then that will be set as the df variable
            df = pd.read_csv(user_file)

    # Display dataset
    st.dataframe(df)

    # Remove rows with missing values
    df.dropna(inplace=True)

    # Define features and target
    st.markdown("""
                ### Important Instructions:
                ###### Depending on the Machine Learning Model you choose to explore, your feature and target variables will change.
                - For Logistic Regression models make sure to select categorical or continuous variables for the features and a binary variable for the target. Binary means the target variable's outcome must be 0 or 1, yes or no.
                 """)
    
    # Choosing features
    features = st.multiselect("Choose features here", options = df.columns) # grab the columns so they have drop down of column names

    # Choosing target variable
    target_var = st.selectbox("Choose your target variable here (target variable cannot be one of selected feature variables)", options = df.columns)
    X = df[features]
    y = df[target_var]
    return df, X, y, features


def split_data(X, y, test_size=0.2, random_state=42): # random state allows for replicated results - improves user experience
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Initialize and train a logistic regression model on unscaled data
def initialize_and_train_logistic_regression():
    # Initialize class (getting it ready)
    log_reg = LogisticRegression()
    # Train our data ('fit' method changes this class automatically)
    log_reg.fit(X_train, y_train)
    # Make predictions on the test set
    y_pred = log_reg.predict(X_test) #this is putting the 20% of our X_test into model & getting what the model predicts the y-value is
    return log_reg, y_pred

# Confusion Matrix
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(plt)
    plt.clf()

### Streamlit App Layout ###

# Selection controls at the top - Create two columns for side-by-side display
st.subheader("Logistic Regression Model")
st.subheader("Data Type")
data_type = st.radio("Choose a type of data:", options=["Unscaled", "Scaled"])

# Load and preprocess the data; split into training and testing sets
df, X, y, features = load_and_preprocess_data()
X_train, X_test, y_train, y_test = split_data(X, y)

# Depending on the toggle, optionally scale the data
if data_type == "Scaled":
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

# Training data and displaying results
col1, col2 = st.columns(2)

with col1:
    st.subheader("Confusion Matrix")
    log_reg, y_pred = initialize_and_train_logistic_regression()
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, "Confusion Matrix for Logistic Regression")

with col2:
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

### Additional Data Information Section ###

st.expander("Click to view Data Information")
st.write("#### First 5 Rows of the Dataset")
st.dataframe(df.head())
st.write("#### Statistical Summary")
st.dataframe(df.describe())

### User Review ###
st.feedback('stars')