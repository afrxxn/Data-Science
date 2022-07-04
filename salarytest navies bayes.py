import pandas as pd
df_train = pd.read_csv("C://Users/NAVEEN REDDY/Downloads/SalaryData_Train.csv")
df1_test = pd.read_csv("C://Users/NAVEEN REDDY/Downloads/SalaryData_Test.csv")
df1_test.shape
list(df1_test)
df1_test.head()

df_train.shape
list(df_train)
df_train.head()

# finding missing values
df1_test.isnull().sum()
df_train.isnull().sum()

#======================================================================

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df_train["Salary_code"] = LE.fit_transform(df_train["Salary"])
df_train[["Salary", "Salary_code"]].head(14)
pd.crosstab(df_train.Salary, df_train.Salary_code)

df_train["native_code"] = LE.fit_transform(df_train["native"])
df_train[["native", "native_code"]].head(14)
pd.crosstab(df_train.native, df_train.native_code)

df_train["workclass_code"] = LE.fit_transform(df_train["native"])
df_train[["workclass", "workclass_code"]].head(14)
pd.crosstab(df_train.workclass, df_train.workclass_code)

df_train["education_code"] = LE.fit_transform(df_train["education"])
df_train[["education", "education_code"]].head(14)
pd.crosstab(df_train.native, df_train.native_code)

df_train["maritalstatus_code"] = LE.fit_transform(df_train["maritalstatus"])
df_train[["maritalstatus", "maritalstatus_code"]].head(14)
pd.crosstab(df_train.maritalstatus,df_train.maritalstatus_code)

df_train["occupation_code"] = LE.fit_transform(df_train["occupation"])
df_train[["occupation", "occupation_code"]].head(14)
pd.crosstab(df_train.occupation, df_train.occupation_code)

df_train["relationship_code"] = LE.fit_transform(df_train["relationship"])
df_train[["relationship", "relationship_code"]].head(14)
pd.crosstab(df_train.relationship, df_train.relationship_code)

df_train["race_code"] = LE.fit_transform(df_train["race"])
df_train[["race", "race_code"]].head(14)
pd.crosstab(df_train.race,df_train.race_code)

df_train["sex_code"] = LE.fit_transform(df_train["sex"])
df_train[["sex", "sex_code"]].head(14)
pd.crosstab(df_train.sex, df_train.sex_code)

df_train.drop(['workclass','education','maritalstatus','occupation','relationship','race','sex','native'], axis=1, inplace=True)
df_train.drop(['Salary'],axis=1,inplace=True)

X_train = df_train.drop(['Salary_code'],axis=1)
y_train = df_train['Salary_code']
#======================================================================

LE = LabelEncoder()
df1_test["native_code"] = LE.fit_transform(df1_test["native"])
df1_test[["native", "native_code"]].head(14)
pd.crosstab(df1_test.native, df1_test.native_code)

df1_test["workclass_code"] = LE.fit_transform(df1_test["native"])
df1_test[["workclass", "workclass_code"]].head(14)
pd.crosstab(df1_test.workclass,df1_test.workclass_code)

df1_test["education_code"] = LE.fit_transform(df1_test["education"])
df1_test[["education", "education_code"]].head(14)
pd.crosstab(df1_test.native,df1_test.native_code)

df1_test["maritalstatus_code"] = LE.fit_transform(df1_test["maritalstatus"])
df1_test[["maritalstatus", "maritalstatus_code"]].head(14)
pd.crosstab(df1_test.maritalstatus,df1_test.maritalstatus_code)

df1_test["occupation_code"] = LE.fit_transform(df1_test["occupation"])
df1_test[["occupation", "occupation_code"]].head(14)
pd.crosstab(df1_test.occupation,df1_test.occupation_code)

df1_test["relationship_code"] = LE.fit_transform(df1_test["relationship"])
df1_test[["relationship", "relationship_code"]].head(14)
pd.crosstab(df1_test.relationship, df1_test.relationship_code)

df1_test["race_code"] = LE.fit_transform(df1_test["race"])
df1_test[["race", "race_code"]].head(14)
pd.crosstab(df1_test.race,df1_test.race_code)

df1_test["sex_code"] = LE.fit_transform(df1_test["sex"])
df1_test[["sex", "sex_code"]].head(14)
pd.crosstab(df1_test.sex, df1_test.sex_code)

df1_test["Salary_code"] = LE.fit_transform(df1_test["Salary"])
df1_test[["Salary", "Salary_code"]].head(14)
pd.crosstab(df1_test.Salary, df1_test.Salary_code)

df1_test.drop(['workclass','education','maritalstatus','occupation','relationship','race','sex','native'], axis=1, inplace=True)
X=df1_test.drop(['Salary'],axis=1)

X_test = df_train.drop(['Salary_code'],axis=1)
y_test = df_train['Salary_code']
#======================================================================
# model development
from sklearn.naive_bayes import MultinomialNB
MNB = MultinomialNB()
MNB.fit(X_train,y_train)
Y_pred = MNB.predict(X_test)
#======================================================================
# confusion matrix and accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(y_test,Y_pred)
acc = accuracy_score(y_test,Y_pred).round(2)

print("naive bayes model accuracy score:" , acc)

''' inference: accuracy score:77% ,so the model predict better'''
