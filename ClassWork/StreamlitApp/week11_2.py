import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------
# Application Information
# -----------------------------------------------
st.title("KNN Performance: Scaled vs. Unscaled")
st.markdown("""
### About This Application
This interactive application demonstrates the performance of a K-Nearest Neighbors (KNN) classifier using the Titanic dataset. You can:
- **Select different numbers of neighbors (k)** to see how it affects model performance.
- **Toggle between unscaled and scaled data** to understand the impact of feature scaling on classification.
- **View performance metrics** including accuracy, a confusion matrix, and a classification report.
The Titanic dataset is preprocessed to include key features like passenger class, age, fare, and a one-hot encoded gender indicator.
""")

# -----------------------------------------------
# Helper Functions
# -----------------------------------------------
def load_and_preprocess_data():
    # Load the Titanic dataset from seaborn
    df = sns.load_dataset('titanic')
    # Remove rows with missing 'age' values
    df.dropna(subset=['age'], inplace=True)
    # One-hot encode the 'sex' column (drop first category)
    df = pd.get_dummies(df, columns=['sex'], drop_first=True)
    # Define features and target
    features = ['pclass', 'age', 'sibsp', 'parch', 'fare', 'sex_male']
    X = df[features]
    y = df['survived']
    return df, X, y, features

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_knn(X_train, y_train, n_neighbors):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(plt)
    plt.clf()

# -----------------------------------------------
# Streamlit App Layout
# -----------------------------------------------

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
    plot_confusion_matrix(cm, f"KNN Confusion Matrix ({data_type} Data)")

with col2:
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

# -----------------------------------------------
# Additional Data Information Section
# -----------------------------------------------
with st.expander("Click to view Data Information"):
    st.write("### Overview of the Titanic Dataset")
    st.write("""
        The Titanic dataset contains information about the passengers aboard the Titanic. 
        It includes details such as passenger class (pclass), age, number of siblings/spouses aboard (sibsp), 
        number of parents/children aboard (parch), fare, and gender (encoded as 'sex_male'). 
        The target variable 'survived' indicates whether the passenger survived.
    """)
    st.write("#### First 5 Rows of the Dataset")
    st.dataframe(df.head())
    st.write("#### Statistical Summary")
    st.dataframe(df.describe())