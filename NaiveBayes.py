import numpy as np

class NaiveBayes:
    def fit(self,X,y):
        n_samples, n_features = X.shape        
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        #init mean, var, priors
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype= np.float64)

        for c, idx in enumerate(self._classes):
            X_c = X[y == c] # all rows where label with label c in y. eg 3 from 0,1,2,3
            self._mean[idx, :] = X_c.mean(axis = 0) # mean of all features of this class along vertical axis. [ind,:] -> means for this index and all columns
            self._var[idx, :] = X_c.var(axis = 0) 
            self._prior = X_c.shape[0]/ float(n_samples)

    def predict(self,X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    def _predict(self, x):
        posteriors = []
        # calculate posterior probability for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx,x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)
        
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        num = np.exp( (x - mean)**2 / (2*var))
        den = np.sqrt(2* np.pi*var)
        return num/den 
    
    # testing

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def accuracy(y_true, y_pred):
        return np.sum(y_true == y_pred)/(len(y_pred))
    
    X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    prediction = nb.predict(X_test)
    print("Accuracy over this dataset: ", accuracy(y_test, prediction))
