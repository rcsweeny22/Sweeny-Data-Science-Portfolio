# Import the Streamlit library
import streamlit as st

# Navigate
# ls (look what is inside of folders)
# cd (you can use cd command and type in geinning of folder name then hit 'tab' and it will bring you there)

# streamlit run Week_3_2 (tab)

# Display a simple text message (on web browser remember to hit 'rerun)
st.title("Hello, streamlit!")
#st.write("This is my first streamlit app!") #could also use markdown instead of write
st.markdown("## This is my first streamlit app!")

# Display a large title on the app

# ------------------------
# INTERACTIVE BUTTON
# ------------------------

# Create a button that users can click.
# If the button is clicked, the message changes.
if st.button("Click me!"):
    st.write("You clicked the button. Nice work!")
else:
    st.write("Go ahead...click the button. I dare you.")

st.selectbox('On a scale from 1(terrible) to 10(great), how are you today?', [1,2,3,4,5,6,7,8,9,10])
st.radio('Pick a farm animal:', ['pig', 'cow', 'horse', 'chicken'])
st.slider('On a scale from 1(no) to 3(easily), can you do a cartwheel?', min_value = 1, max_value = 3)

# ------------------------
# COLOR PICKER WIDGET
# ------------------------

# Creates an interactive color picker where users can choose a color.
# The selected color is stored in the variable 'color'.

# Display the chosen color value

# ------------------------
# ADDING DATA TO STREAMLIT
# ------------------------

# Import pandas for handling tabular data

# Display a section title

# Create a simple Pandas DataFrame with sample data


# Display a descriptive message

# Display the dataframe in an interactive table.
# Users can scroll and sort the data within the table.

# ------------------------
# INTERACTIVE DATA FILTERING
# ------------------------

# Create a dropdown (selectbox) for filtering the DataFrame by city.
# The user selects a city from the unique values in the "City" column.

# Create a filtered DataFrame that only includes rows matching the selected city.

# Display the filtered results with an appropriate heading.
  # Show the filtered table

# ------------------------
# NEXT STEPS & CHALLENGE
# ------------------------

# Play around with more Streamlit widgets or elements by checking the documentation:
# https://docs.streamlit.io/develop/api-reference
# Use the cheat sheet for quick reference:
# https://cheat-sheet.streamlit.app/

### Challenge:
# 1️⃣ Modify the dataframe (add new columns or different data).
# 2️⃣ Add an input box for users to type names and filter results.
# 3️⃣ Make a simple chart using st.bar_chart().