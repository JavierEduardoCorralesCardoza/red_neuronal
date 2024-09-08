import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
dataset = pd.read_csv('2019.csv')

columnas_restantes = ["GDP per capita", "Social support", "Healthy life expectancy", "Freedom to make life choices"]
target = "GDP per capita"

dataset = dataset[columnas_restantes]
dataset = dataset.astype(np.float32)

# Prepare the data
X = dataset.drop(columns=target).values
y = dataset[target].values

print('# of samples:', len(X))

# Split dataset into training and test sets
np.random.seed(44)
indices = np.arange(X.shape[0])
np.random.shuffle(indices)

train_size = int(0.8 * len(indices))
train_indices, test_indices = indices[:train_size], indices[train_size:]

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# Further split training set into training and validation sets
train_size = int(0.8 * len(train_indices))
train_indices, val_indices = train_indices[:train_size], train_indices[train_size:]
X_train, X_val = X[train_indices], X[val_indices]
y_train, y_val = y[train_indices], y[val_indices]

def train_linear_regression(X_train, y_train, X_val, y_val, epochs=1000, learning_rate=0.01, print_interval=200):
    # Add intercept term (bias) to the input features
    X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    X_val_b = np.c_[np.ones((X_val.shape[0], 1)), X_val]
    m_train, n = X_train_b.shape
    
    # Initialize parameters
    theta = np.random.randn(n, 1)
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    
    # To store the error for plotting
    train_error_history = []
    val_error_history = []
    
    for epoch in range(epochs):
        # Compute predictions
        y_train_pred = X_train_b.dot(theta)
        y_val_pred = X_val_b.dot(theta)
        
        # Compute the error
        train_error = y_train_pred - y_train
        val_error = y_val_pred - y_val
        train_cost = (1 / (2 * m_train)) * np.sum(train_error ** 2)
        val_cost = (1 / (2 * X_val_b.shape[0])) * np.sum(val_error ** 2)
        
        train_error_history.append(train_cost)
        val_error_history.append(val_cost)
        
        # Compute gradients
        gradients = (1 / m_train) * X_train_b.T.dot(train_error)
        
        # Update parameters
        theta -= learning_rate * gradients
        
        if epoch % print_interval == 0:
            # Print the current values of the parameters and cost
            w = theta[1:].flatten()  # weights excluding the bias
            b = theta[0]  # bias
            print(f'Epoch {epoch}:')
            print(f'  w: {w}')
            print(f'  b: {b[0]}')
            print(f'  Training Loss: {train_cost:.4f}')
            print(f'  Validation Loss: {val_cost:.4f}')
    
    return theta, train_error_history, val_error_history

def predict(X, theta):
    # Add intercept term (bias) to the input features
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return X_b.dot(theta)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    y_mean = np.mean(y_true)
    total_variance = np.sum((y_true - y_mean) ** 2)
    residual_variance = np.sum((y_true - y_pred) ** 2)
    return 1 - (residual_variance / total_variance)

# Train the model
theta, train_error_history, val_error_history = train_linear_regression(X_train, y_train, X_val, y_val, epochs=6000, learning_rate=0.01, print_interval=200)

# Predict on the test set
y_test_pred = predict(X_test, theta)

# Evaluate the model
mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print(f'Mean Squared Error: {mse:.4f}')
print(f'R^2 Score: {r2:.4f}')

# Convert y_test and y_test_pred to 1D arrays for plotting
y_test = y_test.flatten()
y_test_pred = y_test_pred.flatten()

# Plot true vs predicted values
plt.figure(figsize=(15, 6))
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_test_pred)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values')

# Plot error over epochs for training and validation
plt.subplot(1, 3, 2)
plt.plot(train_error_history, label='Training Error')
plt.plot(val_error_history, label='Validation Error')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('Error vs. Epochs')
plt.legend()

# Plot residuals
residuals = y_test - y_test_pred
plt.subplot(1, 3, 3)
plt.scatter(y_test_pred, residuals)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')

plt.tight_layout()
plt.show()
