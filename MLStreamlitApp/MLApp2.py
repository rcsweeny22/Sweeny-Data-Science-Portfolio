import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

st.title("Machine Learning Application: KNN Performance")
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

def train_knn(X_train, y_train, n_neighbors):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

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

# Selection controls at the top
st.markdown("### Select Parameters")
k = st.slider("Select number of neighbors (k, odd values only)", min_value=1, max_value=21, step=2, value=5)
data_type = st.radio("Data Type", options=["Unscaled", "Scaled"])

# Load and preprocess the data; split into training and testing sets
df, X, y, features = load_and_preprocess_data()
X_train, X_test, y_train, y_test = split_data(X, y)

# Depending on the toggle, optionally scale the data
if data_type == "Scaled":
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

# Train KNN with the selected k value
knn_model = train_knn(X_train, y_train, n_neighbors=k)
if data_type == "Scaled":
    st.write(f"**Scaled Data: KNN (k = {k})**")
else:
    st.write(f"**Unscaled Data: KNN (k = {k})**")

# Predict and evaluate
y_pred = knn_model.predict(X_test)
accuracy_val = accuracy_score(y_test, y_pred)
st.write(f"**Accuracy: {accuracy_val:.2f}**")

# Create two columns for side-by-side display
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