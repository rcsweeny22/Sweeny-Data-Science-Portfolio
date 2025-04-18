import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

tab1, tab2, tab3, tab4 = st.tabs(["General App Information","User Input", "Model Accuracy","Additional Data Information"]) # Organize app into different tabs

with tab1:
    st.title("Machine Learning Application: KNN Performance")
    st.markdown("""
    ### About This Application
    This interactive application demonstrates the different elements of K Nearest-Neighbors (KNN).
    KNN is a classification model that calculates the distance between all training dataset points and the new data point that users want to classify. By identifying the k nearest neighbors, the model is able to assign the new data point a class based on the class of the majority of it's neighbors.
    
    You should use KNN when your target variable is categorical and binary or multi-class. KNN depends upon the idea that data points near one another and with similar features have the same or similar outcomes. When using KNN in the real world, proper scaling is essential.        
    
    In this app, you can:
    - Use one of Seaborn's pre-loaded datasets like Titanic, Penguins, or Taxis, or upload your own csv.file.
    - Input different features and target variables to explore the elements of K Nearest-Neighbors.
    - Toggle between different parameters to change the number of neighbors (k) used to classify the data.
    - Compare between scaled and unscaled data.
    - Calculate the overall accuracy score as well as the F-1 score for each section of the Confusion Matrix.
    """)
    st.error("Warning: You might get an error message until you go to the second tab and input a continuous variable for features.")

### Download or Upload DataSet ###

with tab2:
    def load_and_preprocess_data(): #defining: loading and preprocessing data
        st.markdown("""
                    ### Important Instructions:
                    ###### For KNN, make sure to select continuous numeric variables for the features and a categorical variable for the target.
                    """)
        file = st.radio("Choose a pre-loaded dataset from Seaborn or upload your own csv.file", options = ['Seaborn dataset', 'Upload csv.file']) # creates upload file option on Streamlit
        df = None
        # Option 1: Insert your own dataset
        if file == 'Seaborn dataset': #begin with Seaborn datasets
            dataset_names = ['titanic', 'penguins', 'taxis'] # curate what Seaborn datasets I want to allow users to choose from
            Seaborn_dataset = st.selectbox("Choose a Seaborn dataset:", dataset_names) # Streamlit widget
            if Seaborn_dataset:# if they choose to look at Seaborn dataset, load it to df
                df = sns.load_dataset(Seaborn_dataset) # defining df by Seaborn dataset
        else:
            user_file = st.file_uploader('Upload a csv file', type = 'csv') # use streamlit widget for uploading files - set to only accepting csv
            if user_file: # if the user uploads a file then that will be set as the df variable
                df = pd.read_csv(user_file) # define df by user csv.file if they choose to upload one

        # Display dataset
        if df is not None: # if the df is defined by Seaborn data or user data
            st.dataframe(df) # display chosen dataset
            # Remove rows with missing values
            df.dropna(inplace=True)
        return df

    def features_and_target_data(df, features, target_var): #define features and target variable
        # Define features and target
        if features == None: # require user to input at least on feature
            st.error("Please choose at least one feature.") # give error message if no features are selected
            
        if target_var in features: # if the user accidentally makes one of their features the target variable too
            st.error("Target variable cannot be a selected feature variable.") # give error message
            

    def split_data(X, y, test_size=0.2, random_state=42): # random state allows for replicated results - improves user experience
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def train_knn(X_train, y_train, n_neighbors): # train k nearest neighbors with training data
        if X_train is None:
            return st.write("Please upload a dataset.")
        else:
            knn = KNeighborsClassifier(n_neighbors=n_neighbors) # import KNeighborsClassifier
            knn.fit(X_train, y_train) # fit knn to the data
            return knn

    # Confusion Matrix
    def plot_confusion_matrix(cm, title): # plot the confusion matrix
        plt.figure(figsize=(6,4)) # size of image
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues') #heatmap with shades of blue
        plt.title(title) #title of entire heatmap
        plt.xlabel('Predicted') #title of x axis
        plt.ylabel('Actual') #title of y axis
        st.pyplot(plt)
        plt.clf()

### Streamlit App Layout ###

    # Selection controls at the top
    st.markdown("### Select Parameters")
    k = st.slider("Select the number of neighbors (k, odd numbers only)", min_value=1, max_value=11, step=2, value=5) # add toggle feature for user to select number of neighbors(k)
    data_type = st.radio("Data Type", options=["Unscaled", "Scaled"]) # add radio selection for user to switch between unscaled and scaled data

    # Load and preprocess the data; split into training and testing sets
    df = load_and_preprocess_data()
    if df is not None: # if df has been defined
        st.markdown("### Select Feature and Target Variables")
        
        # Choosing features
        features = st.multiselect("Choose features", options = df.columns) # grab the columns so they have drop down of column names

        # Choosing target variable
        target_var = st.selectbox("Choose the target variable", options = df.columns) # selectbox since you can only have one target variable

        features_and_target_data(df, features, target_var)
        X = df[features]
        y = df[target_var]

        X_train, X_test, y_train, y_test = split_data(X, y) # test, train, and split data

        # Depending on toggle, data can be scaled or unscaled
        if data_type == "Scaled":
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
    else:
        st.write("Please upload a dataset.")

with tab3:
        if X_train is None:
            st.write("Please upload a dataset.")
        else:
            # Train KNN with the selected k value
            knn_model = train_knn(X_train, y_train, n_neighbors=k)
            if data_type == "Scaled":
                st.write(f"**Scaled Data: KNN (k = {k})**")
            else:
                st.write(f"**Unscaled Data: KNN (k = {k})**")

            # Predict and evaluate

            y_pred = knn_model.predict(X_test)
            accuracy_val = accuracy_score(y_test, y_pred) # train accuracy score
            st.write(f"**Accuracy: {accuracy_val:.2f}**")

            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            plot_confusion_matrix(cm, "Confusion Matrix for Logistic Regression")
            st.markdown("""
                        ### Confusion Matricies show the Actual values compared to the Predicted values.
                        - The upper left quadrant has the True Negatives, which means the number of datapoints the model predicts to be negative (0) and in actuality are negative (0). We want this quadrant to be high because that means it is good at correctly classifying negatives.
                        - The upper right quadrant is the False Positives, which means the model predicts a positive (1) outcome but in actuality the data point was negative (0). We do not want this quadrant to have a high number.
                        - The lower left quadrant is False Negatives, which are the points which the model predicts to be negative (0) but are actually positive (1). We want to limit this number as well.
                        - Finally, the lower right quadrant is the True Positives where the model predicts a positive (1) outcome and it is actually positive (1). We want to maximize True Positives and True Negatives because that means the model is good at classifying.
                        """) # explain matix so any user can understand

            # Classification Report
            st.subheader("Classification Report")
            st.text(classification_report(y_test, y_pred))
            st.markdown("""
                        - Precision here is the ratio of correctly predicted classes (True Positives) over the total predicted classes (True Positives + False Positives).
                        - Recall depicts the ratio of correctly predicted classes (True Positives) to all the data in the actual dataset class (True Positives + False Negatives).
                        - F1 Scores take into account precision and recall.
                        - The overall accuracy score gives a solid idea of how good the model is at classifying data.
                        """) # explain the different metrics

### Additional Data Information Section ###

with tab4:
    st.expander("Click to view Data Information")
    st.write("#### First 5 Rows of the Dataset")
    st.dataframe(df.head())
    st.write("#### Statistical Summary")
    st.dataframe(df.describe())

    ### User Review ###
    st.write("Rate this app!")
    st.feedback('stars')