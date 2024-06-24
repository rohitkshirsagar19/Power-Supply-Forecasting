# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = 'Power Supply Dataset.csv'
data = pd.read_csv(file_path)

# Filter out rows where 'Date' is not a valid date
data = data[pd.to_datetime(data['Date'], errors='coerce').notna()]

# Convert the 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'], format='%d-%b-%Y')

# Set the 'Date' column as the index
data.set_index('Date', inplace=True)

# Handle missing values if any
data.ffill(inplace=True)

# Feature engineering: create time-based features
data['year'] = data.index.year
data['month'] = data.index.month
data['day'] = data.index.day
data['day_of_week'] = data.index.dayofweek

# Target variable: Energy Required (MU)
target = 'Energy Required (MU)'

# Features
features = data.columns.drop(target)

# Define the feature matrix X and the target vector y
X = data[features]
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models
rf = RandomForestRegressor(n_estimators=100, random_state=42)
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
lr = LinearRegression()

# Train the models
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)
lr.fit(X_train, y_train)

# Make predictions
rf_pred = rf.predict(X_test)
gb_pred = gb.predict(X_test)
lr_pred = lr.predict(X_test)

# Evaluate the models
rf_mse = mean_squared_error(y_test, rf_pred)
gb_mse = mean_squared_error(y_test, gb_pred)
lr_mse = mean_squared_error(y_test, lr_pred)

print(f'Random Forest MSE: {rf_mse}')
print(f'Gradient Boosting MSE: {gb_mse}')
print(f'Linear Regression MSE: {lr_mse}')

# Combine predictions by averaging
ensemble_pred = (rf_pred + gb_pred + lr_pred) / 3
ensemble_mse = mean_squared_error(y_test, ensemble_pred)

print(f'Ensemble MSE: {ensemble_mse}')

# Visualize the target variable over time
plt.figure(figsize=(12, 6))
plt.plot(data.index, data[target])
plt.title('Energy Required (MU) Over Time')
plt.xlabel('Date')
plt.ylabel('Energy Required (MU)')
plt.show()

# Visualize correlations between features
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.show()

# Plot actual vs. predicted values
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label='Actual', alpha=0.7)
plt.plot(y_test.index, ensemble_pred, label='Ensemble Prediction', alpha=0.7)
plt.title('Actual vs. Predicted Energy Required (MU)')
plt.xlabel('Date')
plt.ylabel('Energy Required (MU)')
plt.legend()
plt.show()

# Define the parameter grid for GridSearchCV
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}

# Perform GridSearchCV for Random Forest
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Get the best model
best_rf = grid_search.best_estimator_

# Make predictions with the best model
best_rf_pred = best_rf.predict(X_test)
best_rf_mse = mean_squared_error(y_test, best_rf_pred)

print(f'Optimized Random Forest MSE: {best_rf_mse}')

# Plot actual vs. optimized predicted values
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label='Actual', alpha=0.7)
plt.plot(y_test.index, best_rf_pred, label='Optimized Random Forest Prediction', alpha=0.7)
plt.title('Actual vs. Optimized Predicted Energy Required (MU)')
plt.xlabel('Date')
plt.ylabel('Energy Required (MU)')
plt.legend()
plt.show()
