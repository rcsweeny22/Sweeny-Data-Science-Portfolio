import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, root_mean_squared_error, r2_score
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
        if user_file: # if the user uploads a file then that will be set as the df variable
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
def lin_reg_fig(X_test, y_test):
    plt.scatter(X_test, y_test, color='blue')
    plt.plot(X_test, lin_reg, color='red')
    plt.title('Linear Regression Model')
    plt.legend
    st.pyplot(plt)
    plt.clf()

# Evaluate Linear Regression Model's Metrics
def lin_reg_metrics(y_test, y_pred):
    rmse_lin = root_mean_squared_error(y_test, y_pred)
    r2_lin = r2_score(y_test, y_pred)
    st.write("Linear Regression Model Metrics:")
    st.write(f"Root Squred Error: {rmse_lin:.2f}")
    st.write(f"R^2 Score: {r2_lin:.2f}")

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
    if not all(pd.api.types.is_numeric_dtype(X[col]) for col in features):
        st.warning("Linear Regression requires all numeric feature variables. Please change your selection.")
    elif not pd.api.types.is_numeric_dtype(y):
        st.warning("Linear Regression requires a numeric target variable. Please change your selection.")
    else:
        lin_reg, y_pred = initialize_and_train_linear_regression()
        lin_reg_fig(lin_reg)
        lin_reg_metrics(y_test, y_pred)

elif selected_model == "Logistic Regression":
    if y.nunique() > 3:
        st.warning("Logistic Regression requires a target variable with 3 or less components. Please change your selection.")
    elif not pd.api.types.is_numeric_dtype(y):
        st.warning("Logistic Regression requires a categorical target variable. Please change your selection.")
    else:
        log_reg, y_pred = initialize_and_train_logistic_regression()
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


# ________________________________________________________________________________________________________________________________________________
# ________________________________________________________________________________________________________________________________________________

#MLAppProject Original Code

import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

st.title("Machine Learning ML Application")
st.markdown("""
### About This Application
This interactive application demonstrates the different elements of a Linear Regression model using either your own data or the Taxi dataset.
You can:
- **Input different numeric, continuous feature variables.
- **Set your own target variable that this machine learning application will predict.
""")

### Download or Upload DataSet ###

def load_and_preprocess_data():
    file = st.radio("Upload your own csv.file or use a Seaborn dataset", options = ['Option 1', 'Option 2'])
    # Option 1: Insert your own dataset
    if file == 'Option 1':
        df = st.file_uploader('Upload a csv file', type = 'csv')
    else:
        df = st.selectbox(sns.load_dataset(['taxis', 'titanic', 'iris', 'penguins'])) # Option 2: Load a dataset from seaborn
    st.dataframe(df)
    # Remove rows with missing values
    df.dropna(inplace=True)
    # Define features and target
    st.markdown("""Since this is a Linear Regression application, make sure to select features and a target variable which are continuous and numeric.
                 """)
    features = st.selectbox("Choose your features here", options = df.columns) # grab the columns so they have drop down of column names
    st.write("The features you selected are", features)
    X = df[features]
    target_var = st.selectbox("Choose your target variable here", options = df.columns)
    y = df[target_var]
    return df, X, y, features


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Initialize and train the linear regression model on unscaled data
def initialize_and_train_linear_regression():
    # Initialize class (getting it ready)
    lin_reg = LinearRegression()
    # Train our data ('fit' method changes this class automatically)
    lin_reg.fit(X_train, y_train)
    # Make predictions on the test set
    y_pred = lin_reg.predict(X_test) #this is putting the 20% of our X_test into model & getting what the model predicts the y-value is NEXT STEP we will compare to y_test
    print(y_pred)


### Streamlit App Layout ###

# Selection controls at the top
data_type = st.radio("Data Type", options=["Unscaled", "Scaled"])

# Load and preprocess the data; split into training and testing sets
df, X, y, features = load_and_preprocess_data()
X_train, X_test, y_train, y_test = split_data(X, y)

# Depending on the toggle, optionally scale the data
if data_type == "Scaled":
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

lin_reg_model = initialize_and_train_linear_regression()
if data_type == "Scaled":
    st.write(f"**Scaled Data: Linear Regression Model**")
else:
    st.write(f"**Unscaled Data: Linear Regression Model**")

# Create two columns for side-by-side display
col1, col2 = st.columns(2)

with col1:
    st.subheader("Trained Linear Regression Model")
    plt.scatter(X_train, y_train, color='blue')
    plt.plot(X_train, lin_reg_model.predict(X_train), color='red')
    plt.title('Linear Regression Model')
    plt.show()

with col2:
    st.subheader("Linear Regression Test Data")
    plt.scatter(X_test, y_test, color='blue')
    plt.plot(X_test, lin_reg_model, color='red')
    plt.title('Linear Regression Model')
    plt.show()

### Additional Data Information Section ###

with st.expander("Click to view Data Information"):
    st.write("#### First 5 Rows of the Dataset")
    st.dataframe(df.head())
    st.write("#### Statistical Summary")
    st.dataframe(df.describe())


#######################################################################################################################################################
#######################################################################################################################################################
#######################################################################################################################################################
#######################################################################################################################################################
#######################################################################################################################################################
#######################################################################################################################################################
# This worked on MLApp2

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
                ###### For Logistic Regression models make sure to select categorical or continuous variables for the features and a categorical variable for the target.
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
log_reg, y_pred = initialize_and_train_logistic_regression()
col1, col2 = st.columns(2)

with col1:
    st.subheader("Confusion Matrix")
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