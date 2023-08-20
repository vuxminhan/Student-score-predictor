# Student Score Prediction 

## About the Data :

**The dataset** The goal is to predict `score` of a student based on based on ~30 features (Regression Analysis).

### Attributes for student-por.csv (Portuguese language course) datasets:
1 `school` - student's school (binary: "GP" - Gabriel Pereira or "MS" - Mousinho da Silveira)\
2 `sex` - student's sex (binary: "F" - female or "M" - male)\
3 `age` - student's age (numeric: from 15 to 22)\
4 `address` - student's home address type (binary: "U" - urban or "R" - rural)\
5 `famsize` - family size (binary: "LE3" - less or equal to 3 or "GT3" - greater than 3)\
6 `Pstatus` - parent's cohabitation status (binary: "T" - living together or "A" - apart)\
7 `Medu` - mother's education (numeric: 0 - none,  1 - primary education (4th grade)\, 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)\
8 `Fedu` - father's education (numeric: 0 - none,  1 - primary education (4th grade)\, 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)\
9 `Mjob` - mother's job (nominal: "teacher", "health" care related, civil "services" (e.g. administrative or police)\, "at_home" or "other")\
10 `Fjob` - father's job (nominal: "teacher", "health" care related, civil "services" (e.g. administrative or police)\, "at_home" or "other")\
11 `reason` - reason to choose this school (nominal: close to "home", school "reputation", "course" preference or "other")\
12 `guardian` - student's guardian (nominal: "mother", "father" or "other")\
13 `traveltime` - home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)\
14 `studytime` - weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)\
15 `failures` - number of past class failures (numeric: n if 1<=n<3, else 4)\
16 `schoolsup` - extra educational support (binary: yes or no)\
17 `famsup` - family educational support (binary: yes or no)\
18 `paid` - extra paid classes within the course subject (Math or Portuguese)\ (binary: yes or no)\
19 `activities` - extra-curricular activities (binary: yes or no)\
20 `nursery` - attended nursery school (binary: yes or no)\
21 `higher` - wants to take higher education (binary: yes or no)\
22 `internet` - Internet access at home (binary: yes or no)\
23 `romantic` - with a romantic relationship (binary: yes or no)\
24 `famrel` - quality of family relationships (numeric: from 1 - very bad to 5 - excellent)\
25 `freetime` - free time after school (numeric: from 1 - very low to 5 - very high)\
26 `goout` - going out with friends (numeric: from 1 - very low to 5 - very high)\
27 `Dalc` - workday alcohol consumption (numeric: from 1 - very low to 5 - very high)\
28 `Walc` - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)\
29 `health` - current health status (numeric: from 1 - very bad to 5 - very good)\
30 `absences` - number of school absences (numeric: from 0 to 93)\
31 `G1` - first period grade (numeric: from 0 to 20)\
31 `G2` - second period grade (numeric: from 0 to 20)\
32 `G3` - final grade (numeric: from 0 to 20, output target)\

Dataset Source Link :
[https://archive.ics.uci.edu/dataset/320/student+performance](https://archive.ics.uci.edu/dataset/320/student+performance)

Research Link :
[https://ieeexplore.ieee.org/document/9222435](https://ieeexplore.ieee.org/document/9222435)

### Screenshot of UI test deployment 

![HomepageUI](./screenshots/home.png)

## Approach for the project 

1. Data Ingestion : 
    * In Data Ingestion phase the data is first read as csv. 
    * Feature engineering measures were taken to improve model accuracy

   1.1. *Categorical Variables Encoding:*
   - Convert categorical variables like "school," "sex," "address," "famsize," "Pstatus," "Mjob," "Fjob," "reason," and "guardian" into       numerical values using techniques like one-hot encoding or label encoding.

   1.2. *Combining Parental Education:*
   - Create a new feature that represents the average of "Medu" (mother's education) and "Fedu" (father's education). This could potentially capture the overall educational background of the student's parents.

   1.3. *Alcohol Consumption:*
   - Combine "Dalc" (workday alcohol consumption) and "Walc" (weekend alcohol consumption) to create a new feature representing the overall alcohol consumption behavior. This might capture the student's attitude towards responsible behavior.

   1.4. *Grade Improvement:*
   - Create a new feature that represents the difference between "G2" and "G1." This could indicate how much a student's grade improved between the first and second periods.

   1.5. *Study Efficiency:*
   - Create a new feature that represents the ratio of "studytime" to "freetime." This could capture how effectively a student utilizes their free time for studying.

   1.6. *Socialization Level:*
   - Create a new feature by combining "goout" (going out with friends) and "freetime." This might represent the student's overall socialization level outside of school.

   1.7. *Total Absences:*
   - Combine "absences" with the number of school days to calculate the student's average daily absences. This might provide a more meaningful measure of absenteeism.

   1.8. *Categorical Interaction:*
   - Create interaction terms between pairs of categorical variables. For example, the combination of "school" and "higher" might indicate how much a student from a particular school aspires for higher education.

   1.9  *Interaction with Internet Access:*
    - Create features that represent interaction terms between "internet" access and other variables like "studytime" or "activities." This could capture how internet access affects study habits and extracurricular activities.
    
   1.10. *Total and average score:*
    - 'G1' 'G2' is combined into total_score and devided by 2 to get average_score
    
   1.11.  *Feature Scaling:*
    - Perform feature scaling (e.g., normalization or standardization) on numeric features like "age," "absences," "G1," "G2," etc. to ensure that all features have comparable scales.
    * Then the data is split into training and testing and saved as csv file.

1. Data Transformation : 
    * In this phase a ColumnTransformer Pipeline is created.
    * for Numeric Variables first SimpleImputer is applied with strategy median , then Standard Scaling is performed on numeric data.
    * for Categorical Variables SimpleImputer is applied with most frequent strategy, then ordinal encoding performed , after this data is scaled with Standard Scaler.
    * This preprocessor is saved as pickle file.

2. Model Training : 
    * In this phase base model is tested . The best model found was catboost regressor.
    * After this hyperparameter tuning is performed on catboost 
    * This model is saved as pickle file.

3. Prediction Pipeline : 
    * This pipeline converts given data into dataframe and has various functions to load pickle files and predict the final results in python.

4. Flask App creation : 
    * Flask app is created with User Interface to predict the gemstone prices inside a Web Application.

## Log
![Log](screenshots/log.png)
## Exploratory Data Analysis Notebook

Link : [EDA](<notebook/1. EDA STUDENT PERFORMANCE .ipynb>)

## Model Training Approach Notebook

Link : [MODEL TRAINING](<notebook/2. MODEL TRAINING.ipynb>)


