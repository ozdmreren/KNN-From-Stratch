import pandas as pd
import numpy as np
from collections import Counter


class KNN:
    def __init__(self,k=3):
        self.k = k
        
    def fit(self,X,y):
        self.X_train = X.to_numpy()
        self.y_train= y.to_numpy()
        
        print(self.X_train)
        print(self.y_train)
        
    def predict(self,x): # x=[8,4]
        y_pred = [self._predict(x)]
        
        return np.array(y_pred)
    
    def _predict(self,x):
        distance = [self._eucledian(x, x_train) for x_train in self.X_train]
        indicies = np.argsort(distance)[:self.k]
        labels = Counter([y_train[index] for index in indicies])
        
        return labels
    
    def _eucledian(self,x1,x2):
        return np.sqrt(np.sum(x1-x2)**2)
        



data = pd.DataFrame(columns=["X1","X2","Y"],data=[[2,4,"KÖTÜ"],[3,6,"İYİ"],[3,4,"İYİ"],[4,10,"KÖTÜ"],[5,8,"KÖTÜ"],[6,3,"İYİ"],[7,9,"İYİ"],[9,7,"KÖTÜ"],[11,7,"KÖTÜ"],[10,2,"KÖTÜ"]])

X_train = data[["X1","X2"]]
y_train = data["Y"]


knn = KNN()
knn.fit(X_train,y_train)
y_pred = knn.predict([3,6])
print(y_pred)