# Slideshow link

https://docs.google.com/presentation/d/1kLVuWWvtw22eIXGVHkY_SOouNGFvt_YNLHfqQn1NTtY/edit?usp=sharing

# Logistic Regression Analysis

### Project Overview
----

This project aims to build a logistic regression model to predict whether the US economy will enter a recession based on various economic indicators. The dataset contains financial and economic variables, and the analysis focuses on cleaning the data, applying feature selection techniques, and evaluating the model's performance using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC. Additionally, hyperparameter tuning is applied to optimize the model.

### Dataset
----
The dataset used for this project is sourced from Kaggle, containing key financial and economic indicators that may contribute to predicting recessions. The dataset includes the following columns:

Price_x
INDPRO
CPI
Interest Rate (3 Mo, 4 Mo, 6 Mo, 1 Yr, 2 Yr, 5 Yr, etc.)
GDP
Rate
BBK_Index
Housing_Index
Recession (Target variable: 1 indicates recession, 0 indicates no recession)

#### Titanic Dataset (Additional Analysis)
----
In addition to the recession analysis, the Titanic dataset was analyzed using similar logistic regression methods, focusing on predicting passenger survival. This served as a secondary experiment for classification modeling.

## Project Structure
----
#### Code Files:

- us_recession_analysis.py: Main script for loading, preprocessing, modeling, and evaluating the US recession prediction model.

- titanic_analysis.py: Code for logistic regression analysis on the Titanic dataset, including feature selection, model evaluation, and hyperparameter tuning.

### Key Steps:
----

#### Data Preprocessing:

Handled missing values and dropped irrelevant columns (e.g., Unnamed: 0).

Applied feature scaling using StandardScaler to normalize the data for better model performance.

#### Feature Selection:

Used Recursive Feature Elimination (RFE) to select the top 10 features contributing to predicting the target variable.

Features selected for the US Recession dataset:

Housing_Index
Price_x
Rate
BBK_Index
10 Yr
1 Yr
INDPRO
2 Yr
6 Mo
20 Yr

#### Model Training:

Trained a logistic regression model using the selected features.
Performed hyperparameter tuning using GridSearchCV to find the best parameters for regularization and solver options.

#### Model Evaluation:

Evaluated the model using accuracy, precision, recall, and F1-score metrics.
Plotted the ROC curve to evaluate the model's discrimination capability.
The best model achieved an AUC score of 0.93, with high precision but lower recall, indicating the model’s strength in identifying true positives but a challenge in detecting all actual recessions.

#### Feature Importance Visualization:

Visualized the importance of each feature based on the logistic regression coefficients. Features with positive coefficients (e.g., Housing_Index, Rate) are positively correlated with recessions, while features with negative coefficients (e.g., Price_x, INDPRO) are negatively correlated.

#### Titanic Dataset Analysis:

Similar steps were followed for the Titanic dataset:
Feature selection using RFE.
Logistic regression modeling.
Model evaluation using accuracy, precision, recall, and F1-score.
Best model achieved an accuracy of 80%, with precision and recall metrics highlighting the model's strengths and weaknesses in predicting passenger survival.

## Results

#### US Recession Prediction:

Best Model Accuracy: 93%
Precision: 100% (indicating no false positives)
Recall: 46% (indicating some false negatives)
F1-Score: 0.63

The logistic regression model showed good precision, meaning the model rarely predicted a recession when there wasn't one, but it did struggle with identifying all actual recessions.

#### Titanic Dataset (Additional Analysis):

Best Model Accuracy: 80%
Precision: 96%
Recall: 71%
F1-Score: 0.82
The Titanic model had a balanced performance, with good precision and a recall indicating it captured the majority of true survival outcomes.


### Conclusion

This project demonstrated how logistic regression can be used for binary classification tasks, with a focus on predicting US recessions based on economic indicators. Although the model achieved good precision, improving recall would make it more effective at identifying recessions in the real world. The additional analysis on the Titanic dataset further confirmed the flexibility of logistic regression for different datasets and problem types.

### Dependencies

The following libraries are required to run the project:

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- PySpark (for the Spark-based analysis)









