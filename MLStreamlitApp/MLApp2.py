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
    file = st.radio("Upload your own csv.file or choose a Seaborn dataset", options = ['Option 1', 'Option 2'])
    dataset_names = sns.get_dataset_names()
    # Option 1: Insert your own dataset
    if file == 'Option 1':
        df = st.file_uploader('Upload a csv file', type = 'csv')
    else:
        df = st.selectbox("Choose a Seaborn dataset:", sns.load_dataset(dataset_names)) # Option 2: Load a dataset from seaborn
    st.dataframe(df)
    # Remove rows with missing values
    df.dropna(inplace=True)
    # Define features and target
    st.markdown("""
                ### Since this is a Linear Regression application, make sure to select features and a target variable which are continuous and numeric.
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

