# Penguins Streamlit App 🐧
>[!IMPORTANT]
>**Skills showcased**: EDA, Pandas, Streamlit, Python

### 🔀 Project Overview:
- This project is a beginner app where users are invited to interract with the Penguins dataset through different filtering parameters. Though it is not shareable through a public link and therefore less user friendly, the instructions will help guide users on how to interract with the dataset.  


### 📄 Instructions:
- Step 1. Go into terminal in VS Code.
- Step 2. Type in: streamlit run basic_streamlit_app/main.py
- Step 3. Use the [CSV file](https://github.com/rcsweeny22/Sweeny-Data-Science-Portfolio/tree/main/basic-streamlit-app/data) in the basic-streamlit-app folder. 
- Step 4. Navigate to the Streamlit App via the link that pops up.
- Step 5. Explore the app by changing the filters and widgets.

### 🍎 App Features:
- This app features different toggles so users can choose different species, islands, bill length, flipper length, and body masses of penguins to interract with on Streamlit. Different widgets such as the select box and slider are encoded for an interractive user experience.
- Here is an example of the code I learned and then used to create this app:
```
bill_length = st.slider("Choose a bill length:", 
                   min_value = df["bill_length_mm"].min(),
                   max_value = df["bill_length_mm"].max())
```


### 🔍 Dataset Description:
- 🐧Seaborn's Penguins dataset has numerous features such as species, island, bill length, bill depth, flipper length, body mass, and sex. You can explore these different variables and filter them in different ways in the app!


### 📉 Results:
- Results will vary based on user selections.


### ✏️ References:
- [Penguins Datasets](https://www.geeksforgeeks.org/seaborn-datasets-for-data-science/#3-penguins-dataset)
- Course slides, notes, homework, and code.
