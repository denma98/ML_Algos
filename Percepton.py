import numpy as np

class Perceptron():
    def __init__(self, lr = 0.001, n_itr = 1000):
        self.lr = lr 
        self.n_itr = n_itr
        self.activation_func = self._unit_step_function
        self.weights = None
        self.bias = None
    
    def fit(self,X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.array([1 if i > 0 else 0 for i in y]) # converting our y to 0s and 1s only

        for _ in range(self.n_itr):
            for idx, X_c in enumerate(X):
                linear_val = np.dot(X_c, self.weights) + self.bias
                y_pred = self._unit_step_function(linear_val) # step function applied on one specific x(x1,x2,...xn) for n features
                update = self.lr *(y_[idx] - y_pred)
                self.weights += update*X_c
                self.bias += update

    def predict(self, X):
        linear = np.dot(X,self.weights) + self.bias
        y_pred = self._unit_step_function(linear) # step function applied on all rows of X
        return y_pred
    
    def _unit_step_function(self, x):
        return np.where(x>=0, 1, 0)
    
    # testing

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    def accuracy(y_pred, y_true):
        return np.sum(y_pred == y_true)/len(y_true)

    X, y = datasets.make_blobs(n_samples=150, n_features=2, centers = 2, cluster_std=1.05, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

    p = Perceptron(lr=0.01, n_itr=1000)
    p.fit(X_train, y_train)
    predicted = p.predict(X_test)
    print("Perceptron classification accuracy", accuracy(y_test, predicted))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)

    x0_1 = np.amin(X_train[:, 0])
    x0_2 = np.amax(X_train[:, 0])

    x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
    x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

    ymin = np.amin(X_train[:, 1])
    ymax = np.amax(X_train[:, 1])
    ax.set_ylim([ymin - 3, ymax + 3])

    plt.show()
