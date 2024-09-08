import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load dataset
dataset = pd.read_csv('2019.csv')

columnas_restantes = ["GDP per capita", "Social support", "Healthy life expectancy", "Freedom to make life choices"]
target = "GDP per capita"

dataset = dataset[columnas_restantes]
dataset = dataset.astype(np.float32)

X = dataset.drop(columns=target)
y = dataset[target]

# Split dataset
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=44)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=44)

# Define model with Dropout and L2 regularization
def set_nn_model_architecture_2():
    model = Sequential()
    model.add(Dense(units=8, input_shape=(X_train.shape[1],), activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=8, activation='relu'))
    model.add(Dense(units=1))  # Output layer
    return model

model_3 = set_nn_model_architecture_2()
model_3.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss='mean_squared_error', metrics=['mean_squared_error'])

# Define early stopping and learning rate reduction
early_stopping = tf.keras.callbacks.EarlyStopping(patience=30, mode="min", restore_best_weights=True)
lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(patience=20, factor=0.005)

# Custom callback to track training and test metrics
class MetricsCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.test_loss = []
        self.test_mean_squared_error = []
    
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        # Calculate and log metrics for the test set
        test_loss, test_mse = self.model.evaluate(X_test, y_test, verbose=0)
        self.test_loss.append(test_loss)
        self.test_mean_squared_error.append(test_mse)

metrics_callback = MetricsCallback()

# Train the model with validation data
training_history_3 = model_3.fit(X_train, y_train, epochs=1000, validation_data=(X_val, y_val), batch_size=40,
                                 callbacks=[early_stopping, lr_reduction, metrics_callback], verbose=0)

# Manually update the history object with test metrics
history = training_history_3.history
history['test_mean_squared_error'] = metrics_callback.test_mean_squared_error
history['test_loss'] = metrics_callback.test_loss

def plot_acc_loss(training_history):
    plt.figure(figsize=(12, 6))
    plt.plot(training_history['mean_squared_error'], label='Training MSE')
    plt.plot(training_history['val_mean_squared_error'], label='Validation MSE')
    plt.plot(training_history['test_mean_squared_error'], label='Test MSE', linestyle='--')
    plt.title('Mean Squared Error vs. Epochs')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

plot_acc_loss(history)

# Predict on all datasets
y_train_pred = model_3.predict(X_train).flatten()
y_val_pred = model_3.predict(X_val).flatten()
y_test_pred = model_3.predict(X_test).flatten()

def r2_score(y_true, y_pred):
    y_mean = np.mean(y_true)
    total_variance = np.sum((y_true - y_mean) ** 2)
    residual_variance = np.sum((y_true - y_pred) ** 2)
    return 1 - (residual_variance / total_variance)

# Plot actual vs predicted values for the test set
def plot_actual_vs_predicted(y_true, y_pred, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, color='blue', label='Predictions')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', label='Ideal fit')
    plt.title(title)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Call the function for the test set
plot_actual_vs_predicted(y_test, y_test_pred, 'Actual vs Predicted (Test Set)')

# Plot the distribution of residuals (errors)
def plot_residuals_distribution(y_true, y_pred):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of Residuals')
    plt.xlabel('Residuals (Errors)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Call the function to plot residuals
plot_residuals_distribution(y_test, y_test_pred)

# Evaluate the model
train_loss, train_mse = model_3.evaluate(X_train, y_train, verbose=0)
val_loss, val_mse = model_3.evaluate(X_val, y_val, verbose=0)
test_loss, test_mse = model_3.evaluate(X_test, y_test, verbose=0)

train_r2 = r2_score(y_train, model_3.predict(X_train).flatten())
val_r2 = r2_score(y_val, model_3.predict(X_val).flatten())
test_r2 = r2_score(y_test, y_test_pred)

print(f'Training Loss: {train_loss:.4f}')
print(f'Training MSE: {train_mse:.4f}')
print(f'Training R^2: {train_r2:.4f}')

print(f'Validation Loss: {val_loss:.4f}')
print(f'Validation MSE: {val_mse:.4f}')
print(f'Validation R^2: {val_r2:.4f}')

print(f'Test Loss: {test_loss:.4f}')
print(f'Test MSE: {test_mse:.4f}')
print(f'Test R^2: {test_r2:.4f}')
