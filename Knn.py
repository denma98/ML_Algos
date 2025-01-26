import numpy as np
from collections import Counter
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:
    def __init__(self, k=3):
        self.k =k    
    def fit(self,X,y):
        self.X_train =X
        self.y_train = y
    def predict(self,X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)
    def _predict(self,x):
        # compute distances
        distances = [euclidean_distance(x, x1) for x1 in self.X_train] # computing distance of this particualr x with all the x in our X (X_train)
        # get k nearest samples, labels
        k_indices = np.argsort(distances)[:self.k] #indices of k nearest samples eg : k= 5, 18, 5, 21, 0, 3 for n samele 0,1,2 .... n-1
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # do a majority vote of 
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    

if __name__ == "__main__":
    from matplotlib.colors import ListedColormap
    from sklearn import datasets
    from sklearn.model_selection import train_test_split 

    # cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

    def accuracy(y_test, y_pred):
        return np.sum(y_test == y_pred)/len(y_test)
    
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2,random_state=42)
    k = 7
    
    arg = KNN(k)
    arg.fit(X_train, y_train)
    predictions = arg.predict(X_test)
    print("KNN classification accuracy is : ", accuracy(y_test,predictions))
