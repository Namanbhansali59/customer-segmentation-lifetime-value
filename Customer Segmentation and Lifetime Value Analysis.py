# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the customer transaction data (replace 'data.csv' with your dataset)
data = pd.read_csv('data.csv')

# Explore the data
print(data.head())

# Feature selection - select relevant features for analysis
features = ['Recency', 'Frequency', 'Monetary']

# Standardize the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[features])

# Perform K-means clustering for customer segmentation
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(data_scaled)

# Visualize the clusters
plt.scatter(data['Frequency'], data['Monetary'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Frequency')
plt.ylabel('Monetary')
plt.title('Customer Segmentation')
plt.show()

# Split the data into training and testing sets for lifetime value prediction
X = data[features]
y = data['Lifetime Value']  # Assuming 'Lifetime Value' is a target variable to predict
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model for predicting lifetime value
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)
