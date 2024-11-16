**Student Performance Analysis!**

This repository contains the analysis and modeling of student performance data. The goal is to explore the relationship between various factors (such as gender, hours studied, resources available, sleep hours, etc.) and their impact on exam scores. The project also explores building predictive models to forecast student performance based on these factors.

Dataset link: https://www.kaggle.com/datasets/lainguyn123/student-performance-factors

Data Overview
The dataset contains several features that represent student-related factors and their corresponding exam scores. The key variables are:

Gender: The gender of the student (Male/Female)

Hours_Studied: The number of hours a student studied for the exam

Previous_Scores: The student's previous scores in similar exams

Attendance: The attendance rate of the student

Access_to_Resources: Whether the student has access to learning resources

Internet_Access: Whether the student has internet access

School_Type: Whether the student attends a public or private school

Sleep_Hours: The number of hours the student sleeps per day

Exam_Score: The final exam score (target variable for regression)

**Steps and Analysis**
Exploratory Data Analysis (EDA)

Missing Data Handling: The dataset is cleaned by dropping missing values and duplicates.

Group Comparisons: Statistical analysis (Z-test) to check if there is a significant difference in exam scores between male and female students.

Visualizations:
Bar Plot: Displays average exam scores by gender.

3D Scatter Plot: Visualizes the relationship between hours studied, previous scores, and exam score.

Histograms: Plots showing how gender affects different attributes like hours studied, attendance, and access to resources.

Line Plot: Exam scores vs. sleep hours.

Heatmap: Correlation matrix of features to identify relationships between variables.

Hypothesis Testing
A Z-test is performed to determine if there is a significant difference in exam scores between male and female students.
Data Preprocessing

Encoding Categorical Variables: Label encoding and one-hot encoding are used to handle categorical data such as gender, school type, and other categorical columns.
Feature Scaling: The features are standardized using StandardScaler to improve model performance.
Modeling

Multiple regression and classification models are used to predict exam scores and school type:
Linear Regression, Ridge Regression, and Lasso Regression are evaluated for continuous target prediction (exam scores).
Random Forest Regressor, Gradient Boosting Regressor, and SVR are also tested.
Model Comparison: Cross-validation is used to compare models, evaluating them based on R² and RMSE scores.
Best Model Selection

Based on the evaluation, Gradient Boosting Regressor was chosen as the best-performing model due to its high R² score of 0.699.
Gradient Boosting Classifier is used to predict whether a student attends a public or private school, with evaluation using accuracy.
Feature Importance

Feature importance from the Gradient Boosting models is visualized to understand which factors are most impactful in predicting exam scores and school type.
Results
Gender Analysis: There is no significant difference between the exam scores of male and female students based on the Z-test.
Model Performance:
Gradient Boosting Regressor: Achieved an R² score of 0.699 for exam score prediction.
Gradient Boosting Classifier: Used to predict whether a student attends a public or private school with high accuracy.
**Conclusion**
The analysis and modeling show that various factors like hours studied, previous scores, attendance, access to resources, and sleep hours are significant in predicting student performance. The Gradient Boosting models performed well in both regression (exam scores) and classification (school type), providing insight into which features most influence student success.

File Structure

/StudentPerformanceAnalysis
    ├── StudentPerformanceFactors.xls  # Dataset used for the analysis
    ├── analysis_notebook.ipynb       # Jupyter notebook for analysis and modeling
    ├── 3d_scatter.png                # 3D scatter plot image
    ├── bar_graph.png                 # Bar plot of average exam scores by gender
    ├── heatmap.png                   # Heatmap of feature correlations
    └── Importance.png                # Feature importance plot from Gradient Boosting
**Requirements**

You will need the following libraries to run the analysis:

pandas
numpy
seaborn
matplotlib
sklearn
scipy
You can install these dependencies via pip:

``pip install pandas numpy seaborn matplotlib scikit-learn scipy``

**Usage**
To run the analysis, simply load the dataset into a Jupyter Notebook and execute the code blocks step by step. Visualizations and model evaluations will be generated for interpretation.
