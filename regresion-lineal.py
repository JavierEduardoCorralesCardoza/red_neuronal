import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load dataset
dataset = pd.read_csv('2019.csv')

columnas_restantes = ["GDP per capita", "Social support", "Healthy life expectancy", "Freedom to make life choices"]
target = "GDP per capita"

dataset = dataset[columnas_restantes]
dataset = dataset.astype(np.float32)

X = dataset.drop(columns=target).values
y = dataset[target].values

# Split dataset
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=44)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=44)

# Add bias term to input features
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_val = np.c_[np.ones(X_val.shape[0]), X_val]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

# Initialize parameters
def initialize_parameters(n_features):
    return np.zeros(n_features)

# Compute the cost function
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    errors = predictions - y
    cost = (1 / (2 * m)) * np.sum(errors ** 2)
    return cost

# Perform gradient descent
def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    cost_history = np.zeros(num_iterations)
    
    for i in range(num_iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        theta -= (learning_rate / m) * X.T.dot(errors)
        cost_history[i] = compute_cost(X, y, theta)
    
    return theta, cost_history

# Hyperparameters
learning_rate = 0.01
num_iterations = 500

# Train the model
theta = initialize_parameters(X_train.shape[1])
theta, cost_history_train = gradient_descent(X_train, y_train, theta, learning_rate, num_iterations)

# Evaluate the model
def evaluate_model(X, y, theta):
    predictions = X.dot(theta)
    errors = predictions - y
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    r2 = 1 - np.sum(errors ** 2) / np.sum((y - np.mean(y)) ** 2)
    return mse, rmse, r2

# Compute metrics for training, validation, and testing datasets
mse_train, rmse_train, r2_train = evaluate_model(X_train, y_train, theta)
mse_val, rmse_val, r2_val = evaluate_model(X_val, y_val, theta)
mse_test, rmse_test, r2_test = evaluate_model(X_test, y_test, theta)

print(f'Training MSE: {mse_train:.4f}')
print(f'Training RMSE: {rmse_train:.4f}')
print(f'Training R^2: {r2_train:.4f}')

print(f'Validation MSE: {mse_val:.4f}')
print(f'Validation RMSE: {rmse_val:.4f}')
print(f'Validation R^2: {r2_val:.4f}')

print(f'Test MSE: {mse_test:.4f}')
print(f'Test RMSE: {rmse_test:.4f}')
print(f'Test R^2: {r2_test:.4f}')

# Plot cost history
plt.figure(figsize=(12, 6))
plt.plot(cost_history_train, label='Training Cost')
plt.title('Cost Function vs. Iterations')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot actual vs predicted values for the test set
def plot_actual_vs_predicted(X, y, theta, title):
    predictions = X.dot(theta)
    plt.figure(figsize=(8, 6))
    plt.scatter(y, predictions, color='blue', label='Predictions')
    plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--', label='Ideal fit')
    plt.title(title)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_actual_vs_predicted(X_test, y_test, theta, 'Actual vs Predicted (Test Set)')

# Plot the distribution of residuals (errors)
def plot_residuals_distribution(X, y, theta):
    predictions = X.dot(theta)
    residuals = y - predictions
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of Residuals')
    plt.xlabel('Residuals (Errors)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_residuals_distribution(X_test, y_test, theta)
