import pandas as pd # import pandas
df = pd.read_csv("C://Users/NAVEEN REDDY/Downloads/Zoo.csv")
df.shape
df.head()
list(df)
df.drop(['animal name'],axis=1,inplace=True) #droping animals because it have 1-99 animals types
# split as X and Y
Y = df["type"]
X = df.iloc[:,1:16]
list(X)

import seaborn as sns
sns.factorplot('type', data=df, kind="count",size = 5,aspect = 2)#As shown in the graphs above, highest number of animals available in Zoo are Type 1 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y ,random_state=42)  # By default test_size=0.25

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, p=1) # k =5 # p=2 --> Eucledian distance
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

#conclusion 
# by applying algorithm of KNN accuracy(96%)