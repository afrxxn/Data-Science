
import pandas as pd
df = pd.read_csv("C://Users/NAVEEN REDDY/Downloads/forestfires.csv")
df.shape
df.head()

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df["size_category_code"] = LE.fit_transform(df["size_category"])
df[["size_category", "size_category_code"]].head(11)
pd.crosstab(df.size_category,df.size_category_code)
df.drop(['size_category'],axis=1,inplace=True)

df["month_code"] = LE.fit_transform(df["month"])
df[["month", "month_code"]].head(11)
pd.crosstab(df.month ,df.month_code)

df["day_code"] = LE.fit_transform(df["day"])
df[["day", "day_code"]].head(11)
pd.crosstab(df.day ,df.day_code)

df.drop(['month','day'],axis=1,inplace=True)

y=df.size_category_code
X = df.drop(['size_category_code'],axis=1)
# Splitting Train and Test
from sklearn.model_selection._split import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=10)

X_train.shape
X_test.shape
y_train.shape
y_test.shape

# Loading SVC 
# Training a classifier - kernel='rbf'
from sklearn.svm import SVC
SVC()
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
y_pred_train=clf.predict(X_train)
# import the metrics class
from sklearn import metrics
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)

print("Training Accuracy :",metrics.accuracy_score(y_train, y_pred_train).round(2))
print("Testing Accuracy:",metrics.accuracy_score(y_test, y_pred).round(2))

cm = metrics.confusion_matrix(y_train, y_pred_train)
print(cm)

'''  conclusion: After fitting the model comparing linear,rbf,ploy . 
  linear is 98% predicting well performing '''
