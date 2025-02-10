import streamlit as st
import pandas as pd

# Creating a title and subtitles
st.title("Analyzing Sample Data: Exploring Palmer's Penguins")
st.subheader("Welcome to my first Streamlit App! This app offers numerous interactive fidgets and buttons which allow the audience to engage with the sample data set I chose, Palmer's Penguins.")
st.markdown("##### Now, let's look at some penguin data!")

# Adding the penguin dataframe
df = pd.read_csv("data/penguins.csv")

# Displaying the table in Streamlit
st.write("Here's the dataset loaded from the sample CSV file:")
st.dataframe(df)

# Species
# Using a selectbox to allow users to filter data by species
species = st.selectbox("Select a species", df["species"].unique())

# Filtering the DataFrame based on user selection
filtered_df = df[df["species"] == species]

# Display the filtered results
st.write(f"Penguins that are of the {species} species:")
st.dataframe(filtered_df)

# Island
# Using a selectbox to allow users to filter data by island
island = st.selectbox("Select an island", df["island"].unique())
# Filtering the DataFrame based on user selection
filtered_df2 = df[df["island"] == island]
# Display the filtered results
st.write(f"Penguins that are from {island} island:")
st.dataframe(filtered_df2)

# Bill Length
# Using a slider to allow users to analyze bill length data
bill_length = st.slider("Choose a bill length:", 
                   min_value = df["bill_length_mm"].min(),
                   max_value = df["bill_length_mm"].max())
# Display the results based on bill length selected by the user
st.write(f"Penguins with bills that are {bill_length} mm:")
st.dataframe(df[df['bill_length_mm'] == bill_length])

st.write(f"Penguins with bills that are smaller than {bill_length} mm:")
st.dataframe(df[df['bill_length_mm'] <= bill_length])

st.write(f"Penguins with bills that are larger than {bill_length} mm:")
st.dataframe(df[df['bill_length_mm'] >= bill_length])

# Flipper Length
# Using a slider to allow users to analyze flipper length data
flipper_length = st.slider("Choose a flipper length:", 
                min_value = df["flipper_length_mm"].min(),
                max_value = df["flipper_length_mm"].max())
# Display the results based on flipper length selected by the user
st.write(f"Penguins with flippers that are {flipper_length} mm:")
st.dataframe(df[df['flipper_length_mm'] == flipper_length])

st.write(f"Penguins with flippers that are smaller than {flipper_length} mm:")
st.dataframe(df[df['flipper_length_mm'] <= flipper_length])

st.write(f"Penguins with flipper that are larger than {flipper_length} mm:")
st.dataframe(df[df['flipper_length_mm'] >= flipper_length])

# Body mass
# Using a slider to allow users to analyze body mass data
body_mass = st.slider("Choose a body mass:", 
                min_value = df["body_mass_g"].min(),
                max_value = df["body_mass_g"].max())
# Display the results based on body mass selected by the user
st.write(f"Penguins with a body mass of {body_mass} g:")
st.dataframe(df[df['body_mass_g'] == body_mass])

st.write(f"Penguins with a body mass that is less than {body_mass} g:")
st.dataframe(df[df['body_mass_g'] <= body_mass])

st.write(f"Penguins with a body mass that is more than {body_mass} g:")
st.dataframe(df[df['body_mass_g'] >= body_mass])

# Asking user to reflect on their enjoyment while using the app
st.select_slider('On a scale from 1 (not enjoyable) to 5 (very enjoyable) how much did you enjoy this app?', options = [1,2,3,4,5])

# Final interactive way for user to feel engaged with the app
if st.button("Click me if you enjoyed this app!"):
    st.write("Thank you so much!")

if st.button("Click me if you did not click the previous button!"):
    st.write("Let me know how I can improve this app!")