from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
# Training function for perceptron and adaline
def train_and_plot_errors(model, X_train, y_train, X_val, y_val, epochs=100):
    train_errors = []
    val_errors = []

    for epoch in range(epochs):
        model.partial_fit(X_train, y_train, classes=np.unique(y_train))
        train_errors.append(1 - model.score(X_train, y_train))
        val_errors.append(1 - model.score(X_val, y_val))

    return train_errors, val_errors

# Generate a two-class dataset with 10,000 data points
X, y = make_classification(n_samples=10000, n_features=2, n_informative=2, n_redundant=0, 
                           n_clusters_per_class=1, class_sep=2.0, random_state=42)

# Plot the distribution of the data
plt.figure(figsize=(10, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')
plt.title('Linearly Separable Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# Initialize the perceptron
perceptron = Perceptron(max_iter=1000, random_state=42)

# Train the perceptron
perceptron.fit(X, y)

# Predict and evaluate
perceptron_accuracy = perceptron.score(X, y)
print(f'Perceptron Accuracy: {perceptron_accuracy:.2f}')

# Initialize Adaline (using SGDClassifier with squared_error loss)
adaline = SGDClassifier(loss='squared_error', max_iter=1000, random_state=42)

# Train the Adaline
adaline.fit(X, y)

# Predict and evaluate
adaline_accuracy = adaline.score(X, y)
print(f'Adaline Accuracy: {adaline_accuracy:.2f}')

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Train perceptron
perceptron = Perceptron(max_iter=1, warm_start=True, random_state=42)
perceptron_train_errors, perceptron_val_errors = train_and_plot_errors(perceptron, X_train, y_train, X_val, y_val)

# Train adaline
adaline = SGDClassifier(loss='squared_error', max_iter=1, warm_start=True, random_state=42)
adaline_train_errors, adaline_val_errors = train_and_plot_errors(adaline, X_train, y_train, X_val, y_val)

# Plot training and validation errors
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(perceptron_train_errors, label='Training Error')
plt.plot(perceptron_val_errors, label='Validation Error')
plt.title('Perceptron Training and Validation Errors')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(adaline_train_errors, label='Training Error')
plt.plot(adaline_val_errors, label='Validation Error')
plt.title('Adaline Training and Validation Errors')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend()

plt.show()
# Final accuracy
print(f'Perceptron Final Accuracy: {1 - perceptron_val_errors[-1]:.2f}')
print(f'Adaline Final Accuracy: {1 - adaline_val_errors[-1]:.2f}')
