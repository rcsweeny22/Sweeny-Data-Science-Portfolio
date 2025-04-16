import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression

st.title("Machine Learning ML Application")
st.markdown("""
### About This Application
This interactive application demonstrates the different elements of Linear and Logistic Regression.
You can:
- Use your own dataset or one of Seaborn's pre-loaded datasets (Iris, Titanic, Penguins, or Taxis).
- Input different numeric, continuous feature and target variables to explore the elements of predictive Linear Regression models.
- Discover binary classification results from Logistic Rregression after selecting categorical and continuous variables for feature and target variables.
""")

### Download or Upload DataSet ###

def load_and_preprocess_data():
    file = st.radio("Upload your own csv.file or choose a pre-loaded dataset from Seaborn", options = ['Upload csv.file', 'Seaborn dataset'])
    # Option 1: Insert your own dataset
    if file == 'Upload csv.file':
        user_file = st.file_uploader('Upload a csv file', type = 'csv') # use streamlit widget for uploading files - set to only accepting csv
        if user_file is not None: # if the user uploads a file then that will be set as the df variable
            df = pd.read_csv(user_file)
    else:
        dataset_names = ['iris', 'titanic', 'penguins', 'taxis'] # curate what Seaborn datasets I want to allow users to choose from
        Seaborn_dataset = st.selectbox("Choose a Seaborn dataset:", dataset_names) # streamlit widget
        if Seaborn_dataset:
            df = sns.load_dataset(Seaborn_dataset)

    # Display dataset
    st.dataframe(df)

    # Remove rows with missing values
    df.dropna(inplace=True)

    # Define features and target
    st.markdown("""
                ### Important Instructions:
                ###### Depending on the Machine Learning Model you choose to explore, your feature and target variables will change.
                - For Linear Regression models, make sure to select feature and target variables which are continuous and numeric.
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

# Initialize and train the linear regression model on unscaled data
def initialize_and_train_linear_regression():
    # Initialize class (getting it ready)
    lin_reg = LinearRegression()
    # Train our data ('fit' method changes this class automatically)
    lin_reg.fit(X_train, y_train)
    # Make predictions on the test set
    y_pred = lin_reg.predict(X_test) #this is putting the 20% of our X_test into model & getting what the model predicts the y-value is
    return lin_reg, y_pred

# Initialize and train a logistic regression model on unscaled data
def initialize_and_train_logistic_regression():
    # Initialize class (getting it ready)
    log_reg = LogisticRegression()
    # Train our data ('fit' method changes this class automatically)
    log_reg.fit(X_train, y_train)
    # Make predictions on the test set
    y_pred = log_reg.predict(X_test) #this is again putting the 20% of our X_test into model & getting what the model predicts the y-value is
    return log_reg, y_pred

# Create a visualization of Linear Regression Model
def lin_reg_fig(lin_reg, X_test, y_test):
    y_pred = lin_reg.predict(X_test)
    plt.scatter(X_test, y_test, color='blue')
    plt.plot(X_test, lin_reg, color='red')
    plt.title('Linear Regression Model')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    st.pyplot(plt)
    plt.clf()

# Evaluate Linear Regression Model's Metrics
def lin_reg_metrics(y_test, y_pred):
    rmse_lin = root_mean_squared_error(y_test, y_pred)
    r2_lin = r2_score(y_test, y_pred)
    print("Unscaled Data Model:")
    print(f"Root Squred Error: {rmse_lin:.2f}")
    print(f"R^2 Score: {r2_lin:.2f}")
    return rmse_lin, r2_lin

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
col1, col2 = st.columns(2)

with col1:
    st.subheader("Regression Model")
    selected_model = st.radio("Choose a type of Regression Model:", options = ["Linear Regression", "Logistic Regression"])

with col2:
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
if selected_model == 'Linear Regression':
    lin_reg, y_pred = initialize_and_train_linear_regression()
    st.subheader("Linear Regression Model")
    lin_reg_fig(lin_reg, X_test, y_test)
    lin_reg_metrics(y_test, y_pred)

elif selected_model == "Logistic Regression":
    log_reg, y_pred = initialize_and_train_logistic_regression()
    st.subheader("Logistic Regression Model")
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, "Confusion Matrix for Logistic Regression")


### Additional Data Information Section ###

with st.expander("Click to view Data Information"):
    st.write("#### First 5 Rows of the Dataset")
    st.dataframe(df.head())
    st.write("#### Statistical Summary")
    st.dataframe(df.describe())

### User Review ###
st.feedback('stars')