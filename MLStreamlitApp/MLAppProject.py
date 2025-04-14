import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

st.title("TITLE")
st.markdown("""
### About This Application
This interactive application demonstrates the Linear Regression model using the Taxi dataset. You can:
- **Input different numbers of passengers or tolls, change the distance and choose different tip amounts to see how the fare changes.
- **View performance metrics** including accuracy, a confusion matrix, and a classification report.
The Taxi dataset is preprocessed to include key features like passengers, distance, tip, and tolls.
""")

### Download or Upload DataSet ###

def load_and_preprocess_data():
    # Load the Taxi dataset from seaborn
    df = sns.load_dataset('taxis')
    # Remove rows with missing 'passengers' values
    df.dropna(subset=['passengers'], inplace=True)
    # Define features and target
    features = ['passengers', 'distance', 'tip', 'tolls']
    X = df[features]
    y = df['fare']
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
st.markdown("### Select Parameters")
passengers = st.slider("Select number of passengers (1 to 6)", min_value=1, max_value=6)

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

### Initialize and train tree classification model ###
model = DecisionTreeClassifier(random_state = 42,
                               max_depth = 4) # Adding this helps prevent overfitting
model.fit(X_train, y_train)


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

