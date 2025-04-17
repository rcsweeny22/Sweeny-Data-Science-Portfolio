# Sweeny Machine Learning Streamlit App

### Project Overview: 🔀
- For this project I dove into K Nearest Neighbors and created a space for users to analyze the accuracy and precision of a classification model. This project is heavily dependent upon proper usage, so users should be sure to read all instructions. My goal was to create something that my parents could use and understand. That is why I have information all throughout my app, so that it is educational as well as entertaining.


### Instructions: 📄
- Step 1. Read all instructions - this app showcases KNN which is best used for classification tasks.
- Step 2. Navigate to the top of the app and find the 4 tabs. Read the 'General App Information' tab.
- Step 3. Select the 'User Input' tab.
  - Step 3(a). Choose the number of k neighbors and the data type. Select a Seaborn dataset or upload a csv.file. Choose features and a target variable.
- Step 4. Navigate to the 'Model Accuracy' tab and explore.
- Step 5. Go to the last tab, 'Additional Data Information.'
- Step 6. Alter the data type and features to see how the accuracy of the model changes.


### App Features: 🍎
- The classification model this app explores is k Nearest Neighbors. KNN is primarily used when the target variable is categorical and binary or multi-class. KNN assumes that data points near one another and with similar features have similar outcomes. KNN is the process by which a machine learns to classify a new datapoint based off the majority class of its number of neighbors (k). When using KNN in the real world, proper scaling is essential but for the sake of this project, users can switch between scaled and unscaled data to see the difference in the model's accuracy score.
- For hyperparameters, this app features a Confusion Matrix and accuracy scores as well as precision and F1 scores. The Confusion Matrix helps show the model's ability to classify true positives, true negatives, false positives, and false negatives. This is helpful because based on what features the user selects, one of these quadrants of the Confusion Matrix could be more heavily weighed. The accuracy score was also highlighted because it gives solid insight into the overall effectiveness the model shows in classifying data.


### Dataset Descriptions: 🔍
- 🛳️The Titanic Seaborn dataset has information about passengers on the Titanic. Variables include age, sex, fare, ticket class, survival status, embarkation port, number of siblings/spouses, and number of a passenger's parents/children on the Titanic. This dataset can be very interesting and helpful to explore, especially when using 'survived' as the target variable.
- 🐧The Penguins dataset is also from Seaborn. It offers helpful data for classification models with its features such as species, island, bill length, bill depth, flipper length, body mass, and sex. You can use the numeric features with a categorical target variable such as species or island to create a model that is able to classify data well!
- 🚗:The Taxi dataset is another Seaborn dataset and it's features include pickup and dropoff date, number of passengers, distance traveled, fare, tip, tolls, total payment, color of taxi, and payment method for taxi rides in a specified time frame. Use different data types to analyze this dataset!
- You can also upload your own dataset and interract with it in this app!


### Results: 📉
- Results will vary based on user selections.
- Example of Confusion Matrix and Dataset:
![Screenshot (110)](https://github.com/user-attachments/assets/2e40971f-4e86-42a3-8754-f91f5b4b5895)


### References: ✏️
- [KNN Machine Learning Article](https://medium.com/@sachinsoni600517/k-nearest-neighbours-introduction-to-machine-learning-algorithms-9dbc9d9fb3b2)
- [Titanic and Penguins Datasets](https://www.geeksforgeeks.org/seaborn-datasets-for-data-science/#3-penguins-dataset)
- [Taxi Dataset](https://www.kaggle.com/datasets/abdmental01/taxis-dataset-yellow-taxi)
- [Amsamms GitHub](https://github.com/Amsamms/General-machine-learning-algorithm/blob/master/main.py)
- [Streamlit Widgets](https://docs.streamlit.io/develop/api-reference/widgets)
- Course slides, notes, homework, and code.
