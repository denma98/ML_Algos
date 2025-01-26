import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets

class LinearRegression:
    def __init__(self, lr=0.001, n_itr=1000):
        self.lr = lr
        self.n_itr = n_itr
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_itr):
            y_pred = np.dot(X, self.weights) + self.bias
            
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Testing
if __name__ == "__main__":
    def mean_squared_error(y_true, y_predicted):
        return np.mean((y_predicted - y_true) ** 2)
    
    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    regressor = LinearRegression(lr=0.01, n_itr=1000)
    regressor.fit(X_train, y_train)

    predictions = regressor.predict(X_test)

    print("The MSE score for our regression model is:", mean_squared_error(y_test, predictions))

    plt.scatter(X_test, y_test, color="blue", label="Test Data")  # Scatter plot of test data
    plt.plot(X_test, predictions, color="red", linewidth=2, label="Regression Line")  # Regression line
    plt.title("Linear Regression on Test Data")
    plt.xlabel("Feature (X)")
    plt.ylabel("Target (y)")
    plt.legend()
    plt.show()
