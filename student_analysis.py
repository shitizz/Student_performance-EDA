#importing all the libraries 
import pandas as pd
import numpy as np 
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier,RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR

#loading the dataframe. 
student = pd.read_csv('StudentPerformanceFactors.xls')
student.info()
#Dropping the null values
student.dropna(inplace=True)
student.drop_duplicates()
student.info()
student.describe()
#hypothesis testing 

#Seperating the groups 
group_M = student[student['Gender'] == "Male"]["Exam_Score"]
group_F = student[student['Gender'] == "Female"]["Exam_Score"]
#Calculating the means and S.d
mean_M, mean_F = group_M.mean(), group_F.mean()
std_M, std_F = group_M.std(), group_F.std()
n_M, n_F = len(group_M), len(group_F)
#Calculating the Z score
pooled_std = np.sqrt((std_M**2 / n_M) + (std_F**2 / n_F))
z_score = (mean_M - mean_F) / pooled_std
# Calculate the p-value for a two-tailed test
p_value = 2 * (1 - norm.cdf(abs(z_score)))
# Interpret the result
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference between the means.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference between the means.")

print("Z-score:", z_score)
print("p-value:", p_value)

average_scores = student.groupby('Gender')['Exam_Score'].mean().reset_index()

# Set the style of the visualization
sns.set(style="whitegrid")

# Create a bar plot
plt.figure(figsize=(8, 5))
sns.barplot(x='Gender', y='Exam_Score', data=average_scores,hue='Gender', palette={'Male': 'blue', 'Female': 'pink'})

# Add titles and labels
plt.title('Average Exam Scores by Gender', fontsize=16)
plt.xlabel('Gender', fontsize=14)
plt.ylabel('Average Exam Score', fontsize=14)
plt.savefig("bar_graph.png",format = 'png')
# Display the plot
plt.show()

#Scatter plot between Hours Studied vs Previous Score vs Exam_Score 
#Adjusting the fig according to pixels
width_inch = 908.6 / 96
height_inch = 1050.1 / 96
#defining the figure size
fig = plt.figure(figsize=(width_inch, height_inch))
#Adding a subplot for the 3d models. 111 means 1 grid shows one col, row. 
ax = fig.add_subplot(111, projection='3d')
# Scatter plot with Exam Score, Previous Scores, and Hours Studied
ax.scatter(student['Exam_Score'], student['Previous_Scores'], student['Hours_Studied'], color='blue', alpha=0.5)
# Labeling the axes
ax.set_xlabel('Exam Score')
ax.set_ylabel('Previous Scores')
ax.set_zlabel('Hours Studied')
# Adding grid (for asthetics)
ax.grid(True)
# Saving the plot for PPT
plt.savefig("3d_scatter.png", format="png", dpi=300, bbox_inches="tight")
plt.show()

#Male vs Female in different attributes and does that affect the exam score

# Create a 2x2 grid for subplots
fig, ax = plt.subplots(2, 2, figsize=(20, 10))
# Create histograms
sns.histplot(data=student, x='Hours_Studied', hue='Gender', ax=ax[0, 0])
sns.histplot(data=student, x='Attendance', hue='Gender', ax=ax[0, 1])
sns.histplot(data=student, x='Access_to_Resources', hue='Gender', ax=ax[1, 0], multiple='dodge')
sns.histplot(data=student, x='Internet_Access', hue='Gender', ax=ax[1, 1], multiple='dodge')
# Set the main title for the figure
plt.suptitle("Data for Hours Spent Learning, Access to Resources, School Attending, and Access to Internet Based on Gender", fontsize=16)
plt.show()


# Create a line plot to show the relationship between Sleep Hours and Exam Score
plt.figure(figsize=(12, 6))
sns.lineplot(x='Sleep_Hours', y='Exam_Score', data=student, marker='o', color='royalblue')

plt.title('Exam Scores vs. Sleep Hours', fontsize=16)
plt.xlabel('Sleep Hours', fontsize=14)
plt.ylabel('Exam Score', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)

plt.tight_layout()

plt.show()

#Making a heatmap to get Corelation coeffiecents
#using LabelEncoder to change categorical data to numeric data 
le = LabelEncoder()
label_encoders = {}
for column in student.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    student[column] = le.fit_transform(student[column])
    label_encoders[column] = le

corr_matrix = student.corr()

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Heatmap of Encoded Categorical and Numeric Features")
#plt.savefig("heatmap.png", format="png", dpi=300, bbox_inches="tight") {if you want to save the graphs}
plt.show()

def prepare_data(student_df):
    # Create dummy variables
    student_encoded = pd.get_dummies(student_df, drop_first=True)
    
    # Prepare features and target
    X = student_encoded.drop(['Exam_Score'], axis=1)
    y = student_encoded['Exam_Score']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X_scaled, y

def evaluate_models(X, y):
    # Define models to test
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf')
    }
    
    # Setup cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Store results
    results = {}
    
    # Evaluate each model
    for name, model in models.items():
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
        mse_scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
        
        results[name] = {
            'R2_mean': cv_scores.mean(),
            'R2_std': cv_scores.std(),
            'RMSE_mean': np.sqrt(mse_scores.mean()),
            'RMSE_std': np.sqrt(mse_scores.std())
        }
        
        print(f"\n{name} Results:")
        print(f"R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        print(f"RMSE: {np.sqrt(mse_scores.mean()):.4f} (+/- {np.sqrt(mse_scores.std())*2:.4f})")
    
    return results

def plot_model_comparison(results):
    # Prepare data for plotting
    model_names = list(results.keys())
    r2_means = [results[model]['R2_mean'] for model in model_names]
    r2_stds = [results[model]['R2_std'] for model in model_names]
    
    # Create plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(model_names, r2_means, yerr=r2_stds, capsize=5)
    plt.title('Model Comparison - R² Scores')
    plt.xlabel('Models')
    plt.ylabel('R² Score')
    plt.xticks(rotation=45, ha='right')  # ha='right' aligns the labels
    plt.tight_layout()  # Automatically adjusts subplot params for better fit
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('bar_compare.png', format='png', bbox_inches='tight', dpi=300)
    plt.show()

def train_best_model(X, y):
    # Train the best performing model (Gradient Boosting in this case)
    best_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    # Fit the model
    best_model.fit(X, y)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=feature_importance,
        x='Importance',
        y='Feature',
        hue='Feature',
        legend=False,
        palette="viridis"
    )
    plt.title("Feature Importance from Gradient Boosting Regressor")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig('Importance.png',format="png")
    plt.show()
    
    return best_model, feature_importance


# Assuming 'student' DataFrame is already loaded
X, y = prepare_data(student)
# Evaluate all models
results = evaluate_models(X, y)
    
# Plot model comparison
plot_model_comparison(results)
    
# Train and analyze best model
best_model, feature_importance = train_best_model(X, y)

student_encoded = pd.get_dummies(student, drop_first=True)

# Define features (X) and target (y) for regression and classification
X = student_encoded.drop(['Exam_Score', 'School_Type'], axis=1)
y_regression = student_encoded['Exam_Score']                # Target for regression (continuous)
y_classification = student_encoded['School_Type']    # Target for classification (binary: 1 for Public, 0 for Private)

# Split the data into train and test sets
X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.2, random_state=42)
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X, y_classification, test_size=0.2, random_state=42)

#Choosing gradient boosting regressor because of high R2 score of 0.699
#1. Gradient Boosting Regressor
gb_reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_reg.fit(X_train, y_train_reg)
y_pred_reg = gb_reg.predict(X_test)

# Evaluate regression model
mse = mean_squared_error(y_test_reg, y_pred_reg)
print("Gradient Boosting Regression Mean Squared Error:", mse)

#2. Gradient Boosting Classifier
gb_cls = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_cls.fit(X_train_cls, y_train_cls)
y_pred_cls = gb_cls.predict(X_test_cls)

# Evaluate classification model
accuracy = accuracy_score(y_test_cls, y_pred_cls)
print("Gradient Boosting Classification Accuracy:", accuracy)

#3. Feature Importance from Gradient Boosting
feature_importances = gb_cls.feature_importances_
features = X.columns

# Sort features by importance
sorted_idx = np.argsort(feature_importances)[::-1]
sorted_features = [features[i] for i in sorted_idx]
sorted_importances = feature_importances[sorted_idx]

# Plot feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x=sorted_importances, y=sorted_features, palette="viridis")
plt.title("Feature Importance from Gradient Boosting Classifier")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()
