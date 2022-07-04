import pandas as pd
df = pd.read_csv("C://Users/NAVEEN REDDY/Downloads/glass.csv")
df.shape
df.head()
list(df)

# split as X and Y
Y = df["Type"]
X = df.iloc[:,1:9]
list(X)
# standardization
from sklearn.preprocessing import StandardScaler
X_scale = StandardScaler().fit_transform(X)
X_scale
type(X_scale)

import seaborn as sns
sns.factorplot('Type', data=df, kind="count",size = 2,aspect = 2) # as shown in figure type 2 is an higher

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X_scale, Y, stratify=Y ,random_state=42)  

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, p=1)
knn.fit(X_train, y_train)
# Prediction
y_pred=knn.predict(X_test)

# Compute confusion matrix
from sklearn import metrics
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)

import numpy as np
print(np.mean(y_pred == y_test).round(3))  
print('Accuracy of KNN with K=5, on the test set: {:.3f}'.format(knn.score(X_test, y_test)))

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred).round(3)
knn.score(X_test, y_test).round(3)

# conclusion 
# by applying KNN algorithm accuracy: 70%
