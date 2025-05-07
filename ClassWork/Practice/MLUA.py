# Option 1: Insert your own dataset
"""

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

        non_numeric_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
        if non_numeric_cols:
            df = pd.get_dummies(df, columns=non_numeric_cols, drop_first=True)

        # Display dataset
        if df is not None: # if the df is defined by Seaborn data or user data
            st.dataframe(df) # display chosen dataset
        return df

        """