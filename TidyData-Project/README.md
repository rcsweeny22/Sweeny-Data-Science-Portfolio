# Sweeny TidyData Project: Travel Back in Time to the 2008 Olympics :running_shirt_with_sash: :mountain_bicyclist: :medal_sports: :weight_lifting: :flying_disc: #

### :earth_africa: Project Overview:
- The goal of this project is to showcase my ability to organize, tidy, manipulate, and control data. Data in the real world is almost never tidy. Whether it be qualitative or quantitative, from a survey or an interview, data is complicated and it is often messy. Therefore, it is imperative for data scientists to have the skills responsible for organizing and cleaning up the real world data which comes across their desks.
- So in this project, I decided to focus on a dataset from the 2008 Olympics in order to highlight my ability to tidy data. I chose this dataset because the Olympics portray sports as more than just a competition, but as an opportunity to represents one's country on the biggest stage and to be a symbol of hardwork and hope to people across the world. Cleaning this dataset felt like a small way I could honor some of the incredible achievements of these atheltes.

### :speaking_head: Instructions:
- Step 1: Download the [olympics_08_medalists.csv](https://github.com/rcsweeny22/Sweeny-Data-Science-Portfolio/blob/main/TidyData-Project/olympics_08_medalists.csv) from the TidyData-Project Folder.
- Step 2: Open Google Colab and copy the code from the [Jupyter Notebook](https://github.com/rcsweeny22/Sweeny-Data-Science-Portfolio/blob/main/TidyData-Project/TidyData-Project.ipynb) in the TidyData-Project Folder.
- Step 3: Ensure that the csv is properly uploaded to Google Colab, either through uploading it or connecting your Google Drive. (If you upload it, you will need to re-upload it each time you go back to the code.)
- Step 4: Run the code.
- Step 5: Compare the original untidy dataset with the tidy dataset.
- Step 6: Analyze the dataset through visualizations and pivot tables.
  
- Dependencies: pandas, seaborn, matplotlib.pyplot
```
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
```
  
### :baby_symbol: Dataset Description:
- Initially, author Giorgio Comai put together a downloadable dataset which measured all medalist from the 2008 Olympics by their place of birth. This is interesting because one would likely assume that the country which an athlete represented in the Olympics was also the place where they were born. This is not always the case though because athletes who compete in the Olympics are still real people with their own unique background. For example, one Olympian from the U.S.A. could have grown up with a parent in the military and while their parent was stationed in France with the rest of the family, they were born. Or, an athlete could have parents from different countries and dual citizenship which permits them to compete and represent a country that they might not have necessarily been born in.
- This data was adapted from it's orginal source though and was made messy and untidy in a specific way in order to be cleaned and tidied in a comprehensible manner. Therefore, before processing the data I created code to make it read-able for my pc and any desired data manipulation purposes.

### 🌟Results:
- After cleaning the messy data I was able to make some interesting visualizations.
![Screenshot (15)](https://github.com/user-attachments/assets/b6337737-542d-4039-95c5-a240a854f521)
- This grouped bar plot visualizes the number of bronze, silver, and gold medals awarded to male and female Olympians. It is interesting that bronze medals have a higher count than silver or gold. Also, male athletes had more medals overall which might indicate that more Olympic Events are gender specific and male-only. 

![Screenshot (17)](https://github.com/user-attachments/assets/267bffda-88b5-4e9d-bd5b-c1bec133f25d)
- This plot checks to ensure that the cleaned dataset did not inaccurately categorize the gender of the medalists in cases of gender specific events like boxing. It also shows that both male and female events are equal which disproves my previous hypothesis about an uneven number of male events.


### :memo: References: 
- I referenced this [Pandas Cheet Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf) and [Tidy Data](https://vita.had.co.nz/papers/tidy-data.pdf) as resources for this TidyData Project.
- For my dataset description, I referenced the [orginal source](https://edjnet.github.io/OlympicsGoNUTS/2008/) of the 2008 Olympic Medalists data that Comai organized.
