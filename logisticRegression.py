import numpy as np

class LogisticRegression:
    def __init__(self, lr= 0.001,n_itr = 1000 ):
        self.lr = lr
        self.n_itr = n_itr
        self.weights = None
        self.bias = None
    def fit(self,X, y):
        n_samples, n_features = X.shape        
        self.weights = np.zeros(n_features)
        self.bias = 0   
        # gradient descent
        for _ in range(self.n_itr):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)
            dw = 1/(n_samples) * np.dot(X.T, (y_pred - y))
            db = 1/(n_samples) * np.sum(y_pred - y)
            self.weights -= self.lr*dw
            self.bias -= self.lr*db
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        predicted = self._sigmoid(linear_model) # find probabilites in range of 0-1
        final_predicted = [1 if i > 0.5 else 0 for i in predicted] # 1 if > 0.5 else its 0
        return np.array(final_predicted)
    
    def _sigmoid(self, x):
        return 1 /(1 + np.exp(-x))
    
# testing

if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    def accuracy(y_true, y_pred):
        return np.sum(y_true == y_pred)/len(y_true)
    
    dc = datasets.load_breast_cancer()
    X, y = dc.data, dc.target

    X_train,X_test, y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=69)
    
    regressor = LogisticRegression(lr=0.001, n_itr=1000)
    regressor.fit(X_train, y_train)
    predicted = regressor.predict(X_test)
    print("Accuracy over breast cancer dataset is:", accuracy(y_test, predicted))
    
