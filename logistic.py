import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("E:\\Downloads\\insurance.csv")

# One-hot encode categorical variables
coded = pd.get_dummies(data)

# Define the feature matrix X and the target vector y
# Assuming 'charges' is a binary target variable for logistic regression
X = coded.drop('charges', axis=1).values.astype(float)
y = (coded['charges'] > coded['charges'].mean()).astype(float)  # Converting 'charges' to binary

# Normalize features (mean normalization)
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X = (X - mean) / std
# print(X)

# Add a column of ones to include the intercept term in the model
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Initialize parameters
theta = np.zeros(X.shape[1], dtype=float)
learning_rate = 0.01
iterations = 1000

# Number of training examples
m = len(y)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function for logistic regression
def compute_cost(X, y, theta):
    h = sigmoid(X.dot(theta))
    cost = -(1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

# Gradient descent for logistic regression
cost_history = []
for i in range(iterations):
    predictions = sigmoid(X.dot(theta))
    errors = predictions - y
    gradient = (1/m) * X.T.dot(errors)
    theta -= learning_rate * gradient
    # Calculate and store the cost
    cost = compute_cost(X, y, theta)
    cost_history.append(cost)

# Predictions
predictions = sigmoid(X.dot(theta))
predicted_classes = (predictions >= 0.5).astype(int)

# Print the resulting parameters
print("Theta:", theta)

# Print the predictions
print("Predictions (probabilities):", predictions)
print("Predicted classes:", predicted_classes)

# Calculate accuracy
accuracy = np.mean(predicted_classes == y) #compairing
print(f"Accuracy: {accuracy}")


# Generate a range of z values
z_values = np.linspace(-10, 10, 200)
sigmoid_values = sigmoid(z_values)

# Plot the sigmoid function
plt.figure(figsize=(12, 6))
plt.plot(z_values, sigmoid_values, 'r')
plt.title('Sigmoid Function')
plt.xlabel('z')
plt.ylabel('sigmoid(z)')
plt.grid(True)
plt.show()


# Plot the cost function history
plt.figure(figsize=(12, 6))
plt.plot(range(iterations), cost_history, 'b')
plt.title('Cost Function History')
plt.xlabel('Number of Iterations')
plt.ylabel('Cost')
plt.show()

# # Scatter plot of actual vs predicted probabilities
# plt.figure(figsize=(12, 6))
# plt.scatter(y, predictions, alpha=0.3, label='Predicted probabilities')
# plt.title('Predicted Probabilities')
# plt.xlabel('actual')
# plt.ylabel('Predicted Probability')
# plt.legend()
# plt.show()
