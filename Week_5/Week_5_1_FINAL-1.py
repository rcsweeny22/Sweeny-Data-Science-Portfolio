import pandas as pd            # Data handling
import seaborn as sns          # Plotting library for statistical data visualization
import matplotlib.pyplot as plt  # Plotting library for custom graphs
import streamlit as st         # Framework for creating interactive web apps

# ================================================================================
# Missing Data & Data Quality Checks
#
# This lecture covers:
# - **Data Validation:** Checking data types, missing values, and basic consistency.
# - **Missing Data Handling:** Options to drop or impute missing data.
# - **Visualization:** Using heatmaps and histograms to understand data distribution.
# ================================================================================
st.title("Missing Data & Data Quality Checks")
st.markdown("""
This lecture covers:
- **Data Validation:** Checking data types, missing values, and basic consistency.
- **Missing Data Handling:** Options to drop or impute missing data.
- **Visualization:** Using heatmaps and histograms to understand data distribution.
""")

# ------------------------------------------------------------------------------
# Load the Dataset
# ------------------------------------------------------------------------------
# Read the Titanic dataset from a CSV file into a pandas DataFrame.
df = pd.read_csv("titanic.csv")

# ------------------------------------------------------------------------------
# Display Summary Statistics
# ------------------------------------------------------------------------------
st.write("**Summary Statistics**")
# Use .describe() to generate statistical summaries (mean, std, count, etc.) and display them.
st.dataframe(df.describe())

# ------------------------------------------------------------------------------
# Check for Missing Values
# ------------------------------------------------------------------------------
st.write("**Number of Missing Values by Column**")
# Compute the sum of missing values per column and display the result.
st.dataframe(df.isnull().sum())

# ------------------------------------------------------------------------------
# Visualize Missing Data with a Heatmap
# ------------------------------------------------------------------------------
st.write("**Heatmap of Missing Values**")
# Create a matplotlib figure and axis for the heatmap.
fig, ax = plt.subplots()
# Plot a heatmap where missing values are highlighted (using the 'viridis' color map, without a color bar).
sns.heatmap(df.isnull(), cmap="viridis", cbar=False)
# Render the heatmap in the Streamlit app.
st.pyplot(fig)

# ================================================================================
# Interactive Missing Data Handling Section
# ================================================================================
st.subheader("Handle Missing Data")

# ------------------------------------------------------------------------------
# User Input: Select Column and Handling Method
# ------------------------------------------------------------------------------
# Let the user select a numeric column to work with.
column = st.selectbox("Choose a column to fill", df.select_dtypes(include=['number']).columns)
# Provide options for how to handle missing data.
method = st.radio("Choose a method", [
    "Original DF", 
    "Drop Rows", 
    "Drop Columns (>50% Missing)", 
    "Impute Mean", 
    "Impute Median", 
    "Impute Zero"
])

# ------------------------------------------------------------------------------
# Prepare a Copy of the DataFrame for Cleaning
# ------------------------------------------------------------------------------
# Create a copy to preserve the original data.
df_clean = df.copy()

# ------------------------------------------------------------------------------
# Apply the Selected Missing Data Handling Method
# ------------------------------------------------------------------------------
if method == "Original DF":
    pass  # Keep the data unchanged.
elif method == "Drop Rows":
    # Remove all rows that contain any missing values.
    df_clean = df_clean.dropna()
elif method == "Drop Columns (>50% Missing)":
    # Drop columns where more than 50% of the values are missing.
    df_clean = df_clean.drop(columns=df_clean.columns[df_clean.isnull().mean() > 0.5])
elif method == "Impute Mean":
    # Replace missing values in the selected column with the column's mean.
    df_clean[column] = df_clean[column].fillna(df[column].mean())
elif method == "Impute Median":
    # Replace missing values in the selected column with the column's median.
    df_clean[column] = df_clean[column].fillna(df[column].median())
elif method == "Impute Zero":
    # Replace missing values in the selected column with zero.
    df_clean[column] = df_clean[column].fillna(0)

# ------------------------------------------------------------------------------
# Side-by-Side Visualization: Original vs. Cleaned Data
# ------------------------------------------------------------------------------
# Create two columns in the Streamlit layout for side-by-side comparison.
col1, col2 = st.columns(2)

# --- Original Data Visualization ---
with col1:
    st.subheader("Original Data Distribution")
    # Plot a histogram (with a KDE) for the selected column from the original DataFrame.
    fig, ax = plt.subplots()
    sns.histplot(df[column].dropna(), kde=True)
    plt.title(f"Original Distribution of {column}")
    st.pyplot(fig)
    st.subheader(f"{column}'s Original Stats")
    # Display statistical summary for the selected column.
    st.write(df[column].describe())

# --- Cleaned Data Visualization ---
with col2:
    st.subheader("Cleaned Data Distribution")
    # Plot a histogram (with a KDE) for the selected column from the cleaned DataFrame.
    fig, ax = plt.subplots()
    sns.histplot(df_clean[column], kde=True)
    plt.title(f"Distribution of {column} after {method}")
    st.pyplot(fig)
    st.subheader(f"{column}'s New Stats")
    # Display statistical summary for the cleaned data.
    st.write(df_clean[column].describe())