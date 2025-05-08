# Import necessary libraries
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


tab1, tab2, tab3, tab4 = st.tabs(["General App Information", #overview of K-Means
                                  "User Input", # input features
                                  "Model Visualization", #PCA and k clusters
                                  "Additional Data Information"]) # Organize app into different tabs

with tab1:
    st.title("Machine Learning Application: K-Means Performance")
    st.markdown("""
    ### About This Application
    Overview:
    - K-Means is a form of clustering that groups points of data together in k number of clusters.
    - It finds the optimal centroid for the k number of clusters and consequently might reveal some helpful information.
    - K-Means Clustering is used to find unclear patterns in unlabeled data, to categorize subgroups within large unlabeled datasets, and as a beginning step to further data science analyses.
    - K-Means is pretty intuitive and easy to implement for large datasets.
                
    Understanding K-Means Clustering:
    - 1. The k number of clusters is randomly chosen.
    - 2. The number of k centroids are randomly given a set of coordinates that each centroid uses to assign each data point based on its distance. Usually Euclidean distance is used. Each data point is therefore assigned a subgroup or a cluster based on the centroid closest to it.
    - 3. The centroids are recalculated based on the mean of each cluster.
    - 4. This process is reiterated until convergence or until a certain number of iterations is complete. Convergence is when the data points in a cluster do not change significantly anymore.
    - 5. This produces k clusters with centroids that “minimize the sum of squared distances” (IDS Week 13_1 slides).

        
    In this app, you can:
    - Use one of Seaborn's pre-loaded datasets like Titanic, Penguins, or Taxis, or upload your own csv.file.
    - Input different features variables to explore the elements of K-Means Clustering.
    - Increase or decrease the number of clusters (k) to see the different PCA 2D results.
    """)
    st.error("Warning: You might get an error message until you go to the 'User Input' tab and select at least one feature variable.")

### Download or Upload DataSet ###

with tab2:
    def preprocess_dataset(df, dropna_columns=None): # preprocessing all data to handle missing rows and to convert categorical variables to numeric
        if dropna_columns:
            df.dropna(subset=dropna_columns, inplace=True) 
        non_numeric_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])] # making every variable is numeric
        if non_numeric_cols:
            df = pd.get_dummies(df, columns=non_numeric_cols, drop_first=True) # this will encode the non-numeric columns
        return df
    
    def load_and_preprocess_data(): #defining: loading and preprocessing data
        st.markdown("""
                    ### Important Instructions:
                    ###### For K-Means Clustering, you have to make sure to scale your data. This has been done for you already, but is important to keep in mind for furture projects.
                    """)
        file = st.radio("Choose a pre-loaded dataset from Seaborn or upload your own csv.file", options = ['Seaborn dataset', 'Upload csv.file'], key = "data_radio") # creates upload file option on Streamlit
        df = None
        # Option 1: Insert your own dataset
        if file == 'Seaborn dataset': #begin with Seaborn datasets
            dataset_names = ['titanic', 'penguins', 'taxis'] # curate what Seaborn datasets I want to allow users to choose from
            Seaborn_dataset = st.selectbox("Choose a Seaborn dataset:", dataset_names) # Streamlit widget
            if Seaborn_dataset == 'titanic':# if they choose to look at Seaborn dataset, load it to df
                df = sns.load_dataset(Seaborn_dataset) # defining df by Seaborn dataset
                df = preprocess_dataset(df, dropna_columns=["age"]) # Remove rows with missing 'age' values
            if Seaborn_dataset == 'penguins':
                df = sns.load_dataset(Seaborn_dataset) # defining df by Seaborn dataset
                df = preprocess_dataset(df, dropna_columns=['sex', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'])
            if Seaborn_dataset == 'taxis':
                df = sns.load_dataset(Seaborn_dataset) # defining df by Seaborn dataset
                df = preprocess_dataset(df, dropna_columns=['payment', 'pickup_zone', 'dropoff_zone', 'pickup_borough', 'dropoff_borough'])
        else:
            user_file = st.file_uploader('Upload a csv file', type = 'csv') # use streamlit widget for uploading files - set to only accepting csv
            if user_file: # if the user uploads a file then that will be set as the df variable
                df = pd.read_csv(user_file) # define df by user csv.file if they choose to upload one
                df.dropna(inplace=True)
                df=preprocess_dataset(df)

        # Display dataset
        if df is not None: # if the df is defined by Seaborn data or user data
            st.dataframe(df) # display chosen dataset
        return df

    def k_means(X_std):
        # Set the number of clusters
        k = st.number_input('Select number of k clusters:', min_value=2, max_value=8) #allow users to input different numbers of k clusters
        kmeans = KMeans(n_clusters = k, random_state=42) #call KMeans function
        clusters = kmeans.fit_predict(X_std) #fit the clusters to KMeans algorithm
        # Output the centroids and first few cluster assignments
        st.write(f"First 15 cluster assignments: {clusters[:15]}")
        return clusters
   
    def pca_viz(X_std, clusters):
        # Reduce the data to 2 dimensions for visualization using PCA
            pca = PCA(n_components=2) # use PCA to reduce dimetionality of large datasets into just 2 dimensions
            X_pca = pca.fit_transform(X_std)


            # Create a scatter plot of the PCA-transformed data, colored by KMeans cluster labels
            plt.figure(figsize=(8, 6)) # size of scatter plot
            for cluster in np.unique(clusters):
                plt.scatter(X_pca[clusters == cluster, 0], X_pca[clusters == cluster, 1],
                            alpha=0.7, edgecolor='k', s=60, label=f'Cluster{cluster}')
            plt.xlabel('Principal Component 1') #label of x-axis
            plt.ylabel('Principal Component 2') #label of y-axis
            plt.title('KMeans Clustering: 2D PCA Projection')
            plt.legend(loc='best')
            plt.grid(True)
            st.pyplot(plt)

    def features_data(df, features): #define features
        # Define features
        if features == None: # require user to input at least on feature
            st.error("Please choose at least one feature.") # give error message if no features are selected

### Streamlit App Layout ###

    # Load and preprocess the data; split into training and testing sets
    df = load_and_preprocess_data()
    if df is not None: # if df has been defined
        st.markdown("### Select Feature Variables")
        # Choosing features
        features = st.multiselect("Choose the feature variables", options = df.columns) # grab the columns so they have drop down of column names
        features_data(df, features)
    if features:
        X = df[features]
        scaler = StandardScaler() #scale data
        X_std = scaler.fit_transform(X) #data must be scaled for K-Means to work
       
    else:
        st.write("Please upload a dataset.")


with tab3:
        st.markdown("""
                    ###### Explore K-Means Clustering:
                        """)
        if X_std is None:
            st.write("Please upload a dataset.")
        else:
            k = st.number_input('Select number of k clusters:', min_value=2, max_value=8) #allow user to input a number for k clusters
            kmeans = KMeans(n_clusters = k, random_state=42)
            clusters = kmeans.fit_predict(X_std)
            if st.button("Visualize Clusters with Principal Component Analysis"):
                pca_viz(X_std, clusters)
        
        st.divider() #create sections between viasualization and text
        
        st.markdown("""
        ###### Understanding the visualization:
        - Although K-Means works well on large datasets it is dependent on the choice of distance metric and the structure of the dataset.
        - This app required a lot of preprocessing steps in order to ensure the dataset structures of different Seaborn and uploaded datasets would be able to work, but K-Means clustering is, in general, dependent on the choice of distance metric and the structure of the dataset.
        - Also, it is helpful to remember that K-Means works best with spherical clusters and can often be influenced by outliers in data.
            """)
        
### Additional Data Information Section ###


with tab4:
    st.expander("Click to view Data Information") #allow user to see basic sttistics of dataset
    st.write("#### First 5 Rows of the Dataset")
    st.dataframe(df.head())
    st.write("#### Statistical Summary")
    st.dataframe(df.describe())

    ### User Review ###
    st.write("Rate this app!") #allow user opportunity to express whether or not they liked the app
    st.feedback('stars')