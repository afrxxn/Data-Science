import pandas as pd 
import numpy as np
df = pd.read_csv("C://Users/NAVEEN REDDY/Downloads/Company_Data.csv")
df.shape
list(df)
df.head()
type(df)

# Converting x variable to number
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['ShelveLoc1'] = LE.fit_transform(df['ShelveLoc'])
df['ShelveLoc1']
df['Urban1'] = LE.fit_transform(df['Urban'])
df['Urban1']
df['US1'] =  LE.fit_transform(df['US'])
df['US1']
list(df)
df.shape

# Drop the variable
df.drop(['ShelveLoc','Urban','US'],axis=1,inplace=True)
df
# Standardization
from sklearn.preprocessing import StandardScaler,LabelEncoder
Scaler = StandardScaler()
x_scale=Scaler.fit_transform(df)
x_scale

x_new = pd.DataFrame(x_scale)
x_new
x = df.iloc[:,1:11]
x.shape
list(x)
x.ndim

y = df.iloc[:,0]
y.shape
y.mean()
y1 = []
for i in range(0,400,1):
    if y.iloc[i,]>=y.mean():
        print('High')
        y1.append('High')
    else:
        print('Low')
        y1.append('Low')
y_new=pd.DataFrame(y1)
y_new=LabelEncoder().fit_transform(y_new)

# Splitting the train and test data sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y_new,test_size=0.25,stratify=y_new,random_state=41)

x_train.shape

x_test.shape


# Selecting the model 
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(max_features=0.4,n_estimators=500,random_state=41)
RF.fit(x_train,y_train)
y_Pred = RF.predict(x)

from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_new,y_Pred)
cm
y_Pred.shape
ac = accuracy_score(y_new,y_Pred)
acscore = (ac*100).round(3)
print("Accuracy score:",acscore)


# Create two lists for training and test errors
train_error = []
test_error = []

# Define a range of ! to 10 (included) neighbours to be tested
settings = np.arange(0.1,1.1,0.1)

# Loop with Decision Tree Classifier through the max depth values to determine the most appropriate (best)
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

for sam_val in settings:
    Classifier = RandomForestClassifier(n_estimators=100,random_state=41,max_features=sam_val)
    Classifier.fit(x_train,y_train)
    
    y_train_pred = Classifier.predict(x_train)
    train_error.append(np.sqrt(metrics.mean_squared_error(y_train_pred, y_train).round(3)))
    
    y_test_pred = Classifier.predict(x_test)
    test_error.append(np.sqrt(metrics.mean_squared_error(y_test_pred, y_test).round(3)))

print(train_error)
print(test_error)

# Visualize results - to help with deciding which n_neigbors yields the best results (n_neighbors=6, in this case)

import matplotlib.pyplot as plt
plt.plot(settings, train_error, label='MSE of the training set')
plt.plot(settings, test_error, label='MSE of the test set')
plt.ylabel('Root Mean Squared Error')
plt.xlabel('Percentage of features in RF')
plt.legend()
import numpy as np
for i in range(1,101,1):
   x_train,x_test,y_train,y_test=train_test_split(x,y_new,stratify=y_new,test_size=0.25,random_state=i)
   print('For random state',i)
   for j in np.arange(0.1,1.1,0.1):
        RFC=RandomForestClassifier(max_features=j,n_estimators=500)
        model=RFC.fit(x_train,y_train)
        y_pred=model.predict(x_test)
        acc=accuracy_score(y_test,y_pred)
        print('Accuracy for max features',j,'is',(acc*100).round(3))
        while acc==90:
            print((acc*100).round(3))
            break

