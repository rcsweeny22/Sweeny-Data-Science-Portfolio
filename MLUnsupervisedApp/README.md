# Unsupervised Machine Learning Streamlit App
>[!IMPORTANT]
>**Skills showcased**: K-Means Clustering, Streamlit, Unsupervised Machine Learning

### Project Overview: 🔀
- In this project I wanted to explore K-Means Clustering through an Unsupervised Machine Learning application. By allowing users to change the number of k clusters and upload their own dataset, this app is clearly interractive. I also really focused on including helpful explnantions throughout the app to explain the functionality and purpose of K-Means Clustering so that all users feel engaged. My goal was to build off of my skills developed in previous projects which I was able to do through the preprocessing steps and my different choices within Streamlit for ease of use.


### Instructions: 📄
- Step 1. Read all general information
  - Note: this app might have an error message until a user inputs at least one feature variable.
- Step 2. Navigate to the top of the page and select the second tab titled 'User Input.'
- Step 3. Select a Seaborn dataset or upload a csv.file. Choose feature variables.
- Step 4. Navigate to the 'Model Visualization' tab and input k number of clusters. Press the button.
- Step 5. Explore the visualization by changing features or k. 
- Step 6. Go to the last tab, 'Additional Data Information.' Make sure to rate the app!


### App Features: 🍎
- This app explores K-Means, a form of clustering that groups points of data together in k number of clusters. K-Means is a helpful method of unsupervised machine learning because it finds the optimal centroid for the k number of clusters and consequently can reveal helpful information such as hidden patterns or new subgroups within large unlabeled datasets.
- Since datasets often have high dimensionality, I use Principal Component Analysis (PCA) in order to reduce the datasets to 2 dimensions. This then allows users to have a visualization in the form of a scatter plot that depicts the K-Means clustered data.


### Dataset Descriptions: 🔍
- 🛳️Seaborn's Titanic dataset has information about passengers on the Titanic. There are numerous variables that users can select as features such as age, sex, fare, ticket class, survival status, embarkation port, number of siblings/spouses, and number of a passenger's parents/children on the Titanic.
- 🐧Seaborn's Penguins dataset has numerous features such as species, island, bill length, bill depth, flipper length, body mass, and sex. You can use these different variables to find subgroups within the dataset!
- 🚗:The Taxi dataset is another Seaborn dataset and it's variables include pickup and dropoff date, number of passengers, distance traveled, fare, tip, tolls, total payment, color of taxi, and payment method for taxi rides in a specified time frame.
- You can also upload your own dataset in the app!


### Results: 📉
- Results will vary based on user selections.
- Example of K-Means Clustering PCA Projection:
![Screenshot (137)](https://github.com/user-attachments/assets/b45d9566-6680-4d96-827e-510cfa70f13a)
- This screenshot is a possible result after clustering the Penguins dataset based on bill length, bill depth, and body mass and setting k=3.


### References: ✏️
- [Titanic and Penguins Datasets](https://www.geeksforgeeks.org/seaborn-datasets-for-data-science/#3-penguins-dataset)
- [Taxi Dataset](https://www.kaggle.com/datasets/abdmental01/taxis-dataset-yellow-taxi)
- [Amsamms GitHub](https://github.com/Amsamms/General-machine-learning-algorithm/blob/master/main.py)
- [Streamlit Widgets](https://docs.streamlit.io/develop/api-reference/widgets)
- Course slides, notes, homework, and code.
