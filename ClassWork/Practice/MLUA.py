import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Organize app into different tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "General App Information",
    "User Input",
    "Clustering and PCA Visualization",
    "Additional Data Information"
])

with tab1:
    st.title("Machine Learning Application: K-Means Clustering")
    st.markdown("""
    ### About This Application
    Summary:
    - Use one of Seaborn's pre-loaded datasets (Titanic, Penguins, or Taxis) or upload your own `.csv` file.
    - Explore K-Means clustering with customizable features and cluster counts.
    - Visualize results with PCA (Principal Component Analysis).
    """)
    st.error("Warning: Make sure to select continuous variables for features.")

with tab2:
    def load_and_preprocess_data():
        st.markdown("### Choose Dataset")
        file_option = st.radio("Select dataset source:", ['Seaborn dataset', 'Upload CSV'])
        df = None

        if file_option == 'Seaborn dataset':
            dataset_names = ['titanic', 'penguins', 'taxis']
            dataset_choice = st.selectbox("Choose a dataset:", dataset_names)
            df = sns.load_dataset(dataset_choice)

            if dataset_choice == 'titanic':
                df.dropna(subset=['age'], inplace=True)
                df = pd.get_dummies(df, columns=['sex'], drop_first=True)
            elif dataset_choice == 'penguins':
                df.dropna(inplace=True)
                df = pd.get_dummies(df, columns=['sex'], drop_first=True)
            elif dataset_choice == 'taxis':
                df.dropna(inplace=True)
                df = pd.get_dummies(df, columns=['payment'], drop_first=True)
        else:
            uploaded_file = st.file_uploader("Upload your CSV file", type='csv')
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                df.dropna(inplace=True)

        if df is not None:
            st.write("### Dataset Preview")
            st.dataframe(df.head())
        return df

    df = load_and_preprocess_data()

    if df is not None:
        st.markdown("### Select Features and Target")
        features = st.multiselect("Select feature columns:", df.columns)
        target = st.selectbox("Select target column:", df.columns)

        if features and target not in features:
            X = df[features]
            st.write("### Features Preview")
            st.dataframe(X.head())
        else:
            st.error("Select valid features and target variables.")

with tab3:
    st.markdown("### K-Means Clustering and PCA")
    if df is not None and features:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[features])

        n_clusters = st.slider("Select number of clusters (k):", 2, 8, value=3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        st.write(f"Cluster Assignments: {clusters[:15]}")

        # PCA Visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        plt.figure(figsize=(8, 6))
        for cluster_label in np.unique(clusters):
            indices = np.where(clusters == cluster_label)
            plt.scatter(X_pca[indices, 0], X_pca[indices, 1], label=f"Cluster {cluster_label}")

        plt.title("PCA Visualization of Clusters")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend()
        st.pyplot(plt)

with tab4:
    if df is not None:
        st.write("### Additional Dataset Information")
        st.dataframe(df.describe())