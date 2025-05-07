import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


tab1, tab2, tab3, tab4 = st.tabs(["General App Information",
                                  "User Input",
                                  "Model Accuracy",
                                  "Additional Data Information"]) # Organize app into different tabs

with tab1:
    st.title("Machine Learning Application: K-Means Performance")
    st.markdown("""
    ### About This Application
    Summary
        
    In this app, you can:
    - Use one of Seaborn's pre-loaded datasets like Titanic, Penguins, or Taxis, or upload your own csv.file.
    - Input different features and target variables to explore the elements of K-Means Clustering.
    - Toggle between different parameters to change the number of clusters (k).
    """)
    st.error("Warning: You might get an error message until you go to the second tab and input a continuous variable for features.")

### Download or Upload DataSet ###

with tab2:
    def load_and_preprocess_data(): #defining: loading and preprocessing data
        st.markdown("""
                    ### Important Instructions:
                    ###### For KNN, make sure to select continuous numeric variables for the features and a categorical variable for the target.
                    """)
        file = st.radio("Choose a pre-loaded dataset from Seaborn or upload your own csv.file", options = ['Seaborn dataset', 'Upload csv.file'], key = "data_radio") # creates upload file option on Streamlit
        df = None
        # Option 1: Insert your own dataset
        if file == 'Seaborn dataset': #begin with Seaborn datasets
            dataset_names = ['titanic', 'penguins', 'taxis'] # curate what Seaborn datasets I want to allow users to choose from
            Seaborn_dataset = st.selectbox("Choose a Seaborn dataset:", dataset_names) # Streamlit widget
            if Seaborn_dataset == 'titanic':# if they choose to look at Seaborn dataset, load it to df
                df = sns.load_dataset(Seaborn_dataset) # defining df by Seaborn dataset
                 # Remove rows with missing 'age' values
                df.dropna(subset=['age'], inplace=True)
                # One-hot encode the 'sex' column (drop first category)
                df = pd.get_dummies(df, columns=['sex'], drop_first=True)
            if Seaborn_dataset == 'penguins':
                df = sns.load_dataset(Seaborn_dataset) # defining df by Seaborn dataset
                df.dropna(subset = ['sex', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'])
                df = pd.get_dummies(df, columns=['sex'], drop_first=True)
            if Seaborn_dataset == 'taxis':
                df = sns.load_dataset(Seaborn_dataset) # defining df by Seaborn dataset
                df.dropna(subset = ['payment', 'pickup_zone', 'dropoff_zone', 'pickup_borough', 'dropoff_borough'])
                df = pd.get_dummies(df, columns=['payment', 'pickup_zone', 'dropoff_zone', 'pickup_borough', 'dropoff_borough'], drop_first=True)
        else:
            user_file = st.file_uploader('Upload a csv file', type = 'csv') # use streamlit widget for uploading files - set to only accepting csv
            if user_file: # if the user uploads a file then that will be set as the df variable
                df = pd.read_csv(user_file) # define df by user csv.file if they choose to upload one
                df.dropna(inplace=True)


        # Display dataset
        if df is not None: # if the df is defined by Seaborn data or user data
            st.dataframe(df) # display chosen dataset
        return df
        
    def features_and_target_data(df, features, target_var): #define features and target variable
        # Define features and target
        if features == None: # require user to input at least on feature
            st.error("Please choose at least one feature.") # give error message if no features are selected
           
        if target_var in features: # if the user accidentally makes one of their features the target variable too
            st.error("Target variable cannot be a selected feature variable.") # give error message


    def k_means():
        # Set the number of clusters
        k = st.number_imput('Select number of k clusters:', min_value=2, max_value=8)
        kmeans = KMeans(n_clusters = k, random_state=42)
        clusters = kmeans.fit_predict(X_std)
        # Output the centroids and first few cluster assignments
        st.write(clusters[:15])
   
    def pca_viz(pca, X_pca):
        # Reduce the data to 2 dimensions for visualization using PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_std)


            # Create a scatter plot of the PCA-transformed data, colored by KMeans cluster labels
            plt.figure(figsize=(8, 6))
            plt.scatter(X_pca[clusters == 0, 0], X_pca[clusters == 0, 1],
                        c='navy', alpha=0.7, edgecolor='k', s=60, label='Cluster 0')
            plt.scatter(X_pca[clusters == 1, 0], X_pca[clusters == 1, 1],
                        c='darkorange', alpha=0.7, edgecolor='k', s=60, label='Cluster 1')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.title('KMeans Clustering: 2D PCA Projection')
            plt.legend(loc='best')
            plt.grid(True)
            plt.show()


### Streamlit App Layout ###

    # Load and preprocess the data; split into training and testing sets
    df = load_and_preprocess_data()
    if df is not None: # if df has been defined
        st.markdown("### Select Feature and Target Variables")
       
        # Choosing features
        features = st.multiselect("Choose the feature variables", options = df.columns) # grab the columns so they have drop down of column names


        # Choosing target variable
        target_var = st.selectbox("Choose the target variable", options = df.columns) # selectbox since you can only have one target variable


        features_and_target_data(df, features, target_var)
        X = df[features]
        y = df[target_var]


        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)
       
    else:
        st.write("Please upload a dataset.")


with tab3:
        if X_std is None:
            st.write("Please upload a dataset.")
        else:
            k = st.number_imput('Select number of k clusters:', min_value=2, max_value=8)
            kmeans = KMeans(n_clusters = k, random_state=42)
            clusters = kmeans.fit_predict(X_std)
            accuracy_score(y, clusters)
            st.write(f"**Accuracy: {accuracy_score:.2f}**")
            st.write(f"** Classification Report: {classification_report(y, clusters)}**")
            print(pca_viz)


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