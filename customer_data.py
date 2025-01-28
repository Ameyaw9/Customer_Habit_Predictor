# # Import necessary libraries
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# # Load the dataset
# data = pd.read_csv("online_shoppers_intention.csv")

# # Data Exploration
# # Overview of dataset
# print("Dataset Overview:")
# print(data.head())

# print("\nDataset Info:")
# print(data.info())

# print("\nSummary Statistics:")
# print(data.describe())

# # Check for missing values
# missing_values = data.isnull().sum()
# print("\nMissing Values:")
# print(missing_values)

# # Distribution of Revenue
# plt.figure(figsize=(6, 4))
# sns.countplot(data=data, x='Revenue', palette='Set2')
# plt.title('Revenue Distribution')
# plt.xlabel('Revenue')
# plt.ylabel('Count')
# plt.show()

# # Distribution of VisitorType with respect to Revenue
# plt.figure(figsize=(8, 5))
# sns.countplot(data=data, x='VisitorType', hue='Revenue', palette='coolwarm')
# plt.title('Revenue Distribution by Visitor Type')
# plt.xlabel('Visitor Type')
# plt.ylabel('Count')
# plt.legend(title='Revenue')
# plt.show()

# # Revenue by Month
# plt.figure(figsize=(12, 6))
# sns.countplot(data=data, x='Month', hue='Revenue', palette='viridis', order=[
#     'Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
# ])
# plt.title('Revenue by Month')
# plt.xlabel('Month')
# plt.ylabel('Count')
# plt.legend(title='Revenue')
# plt.show()

# # Weekend vs Revenue
# plt.figure(figsize=(8, 5))
# sns.countplot(data=data, x='Weekend', hue='Revenue', palette='Set2')
# plt.title('Revenue Distribution by Weekend')
# plt.xlabel('Weekend')
# plt.ylabel('Count')
# plt.legend(title='Revenue')
# plt.show()

# # Traffic Type vs Revenue
# plt.figure(figsize=(14, 6))
# sns.countplot(data=data, x='TrafficType', hue='Revenue', palette='coolwarm')
# plt.title('Revenue Distribution by Traffic Type')
# plt.xlabel('Traffic Type')
# plt.ylabel('Count')
# plt.legend(title='Revenue')
# plt.show()

# # Preprocessing
# # Encode categorical variables
# encoded_data = data.copy()
# label_encoders = {}
# categorical_columns = ['Month', 'VisitorType', 'Weekend']

# for column in categorical_columns:
#     le = LabelEncoder()
#     encoded_data[column] = le.fit_transform(encoded_data[column])
#     label_encoders[column] = le

# # Separate features and target variable
# X = encoded_data.drop('Revenue', axis=1)
# y = encoded_data['Revenue']

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# # Standardize numerical features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Build a Random Forest model
# rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
# rf_model.fit(X_train_scaled, y_train)

# # Predictions
# y_pred = rf_model.predict(X_test_scaled)

# # Evaluate the model
# conf_matrix = confusion_matrix(y_test, y_pred)
# class_report = classification_report(y_test, y_pred)
# accuracy = accuracy_score(y_test, y_pred)

# # Display results
# print("Confusion Matrix:")
# print(conf_matrix)

# print("\nClassification Report:")
# print(class_report)

# print("\nAccuracy Score:")
# print(accuracy)



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Replace Google Colab file upload with local file input
file_path = input("Enter the path to the CSV file: ")  # Prompt user for file path
df2 = pd.read_csv(file_path)  # Read the CSV file from the specified path

# Step 3: Dataset overview
print("Dataset Size:", df2.shape)
print("First 10 Records:\n", df2.head(10))
print("Summary Statistics:\n", df2.describe())
print("Missing Values:\n", df2.isnull().sum())

# Step 4: Visualizations
sns.set(style="whitegrid")

# Revenue Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x="Revenue", data=df2, palette='Set2')
plt.title("Revenue Distribution")
plt.show()

# VisitorType Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x="VisitorType", data=df2, palette='coolwarm')
plt.title("VisitorType Distribution")
plt.show()

# Step 5: Revenue by TrafficType
x, y = 'TrafficType', 'Revenue'
df1 = df2.groupby(x)[y].value_counts(normalize=True).mul(100).rename('percent').reset_index()
sns.catplot(x=x, y='percent', hue=y, kind='bar', data=df1, height=5, aspect=2)
plt.title("Revenue Distribution by TrafficType")
plt.show()

# Step 6: Cluster Analysis
# KMeans clustering for 'Administrative Duration' and 'Bounce Rate'
x = df2.iloc[:, [1, 6]].values
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

# Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Apply KMeans with optimal clusters
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(x)

# Visualize the clusters
plt.figure(figsize=(8, 5))
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='yellow', label='Centroids')
plt.title('Administrative Duration vs Bounce Rate Clusters')
plt.xlabel('Administrative Duration')
plt.ylabel('Bounce Rate')
plt.legend()
plt.show()

# Step 7: Predictive Modeling (Random Forest & Logistic Regression)
# Preprocessing (Encode categorical features)
df2.fillna(0, inplace=True)
categorical_columns = ['Month', 'VisitorType', 'Weekend']
le = LabelEncoder()
for col in categorical_columns:
    df2[col] = le.fit_transform(df2[col])

# Feature-target split
X = df2.drop(columns=['Revenue'])
y = df2['Revenue']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Random Forest Classifier Results:")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# Logistic Regression Classifier
lr_model = LogisticRegression(solver='liblinear', random_state=42)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("Logistic Regression Classifier Results:")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("Classification Report:\n", classification_report(y_test, y_pred_lr))

# Step 8: Compare Models using ROC Curve
# Random Forest ROC Curve
y_pred_prob = rf_model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Save Predictions
rf_predictions = pd.DataFrame({'Predicted_Revenue_RF': y_pred_rf})
lr_predictions = pd.DataFrame({'Predicted_Revenue_LR': y_pred_lr})

print("Predictions from Random Forest:\n", rf_predictions.head())
print("Predictions from Logistic Regression:\n", lr_predictions.head())
