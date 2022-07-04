
#Logistic Regression
import pandas as pd
data=pd.read_csv("F:\\ExcelR\\Assignments\\Logistic regression\\bank-full.csv",sep=";")
data.shape
data.head()
list(data)
X=data.iloc[:,0:16]
X.head()
list(X)
X.dtypes

#Considering Categorical variables in X1
X1=data[['job','marital','education','default','housing','loan','contact','month','poutcome']]
X1.head()
type(X1)

#Converting to Numeric format
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
for i in range(0,9,1):
    X1.iloc[:,i]=LE.fit_transform(X1.iloc[:,i])
print(X1)

#Considering Numerical variables in X2
X2=data[['age','balance','day','duration','campaign','pdays','previous']]
X2.head()
X2.hist()

#Standardizing the numerical data
from sklearn.preprocessing import StandardScaler
X2_new=StandardScaler().fit_transform(X2)
X2_new=pd.DataFrame(X2_new)
X2_new.set_axis(['age','balance','day','duration','campaign','pdays','previous'],axis='columns',inplace=True)

#Concatinating both numerical and categorical data into new dataset
X_new=pd.concat([X1,X2_new],axis=1)
X_new.head()
list(X_new)

#Target variable description
Y=data['y']
Y.value_counts()
#Converting Y to numeric format
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
Y=LE.fit_transform(Y)

#Importing Logistic Regression
#Fitting the model (Trying to fit first only with numerical variables in X i.e., X2_new which is standardized)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression().fit(X2_new,Y)
Y_pred=model.predict(X2_new)

#Metrics
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score
cm=confusion_matrix(Y,Y_pred)
acc=accuracy_score(Y,Y_pred)
rec=recall_score(Y,Y_pred)
f1=f1_score(Y,Y_pred)

print('Confusion matrix:',cm)
print('Accuracy Score:',(acc*100).round(3))
print('Recall Score:',(rec*100).round(3))
print('F1 Score:',(f1*100).round(3))

#-----------------------------------------------------------------------------
#Fitting the model (Both categorical and numerical variables in X i.e.,X_new)
from sklearn.linear_model import LogisticRegression
model1=LogisticRegression().fit(X_new,Y)
Y_pred1=model1.predict(X_new)

#Metrics
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score
cm1=confusion_matrix(Y,Y_pred1)
acc1=accuracy_score(Y,Y_pred1)
rec1=recall_score(Y,Y_pred1)
f11=f1_score(Y,Y_pred1)

print('Confusion matrix:',cm1)
print('Accuracy Score:',(acc1*100).round(3))
print('Recall Score:',(rec1*100).round(3))
print('F1 Score:',(f11*100).round(3))

'''
#Inference: After including all the categorical variables of X into the model fitting,
            the accuracy just increased from 88% to 89% which is not much significant
            rather we can drop those categorical variables of X and fitting the model
            which will reduce the model complexity.
'''
#-----------------------------------------------------------------------------
#Fitting the model from statsmodels package
import statsmodels.api as sma
model2=sma.Logit(Y,X_new).fit()
model2.summary()

'''
#Inference: From the summary of the statsmodels package we can observe that p value of 
            'education' column is more than 0.05, so checking out the model performance
            by dropping that variable.
'''

#-----------------------------------------------------------------------------
X_new1=X_new.drop(['education'],axis=1)

#Fitting the model
from sklearn.linear_model import LogisticRegression
model2=LogisticRegression().fit(X_new1,Y)
Y_pred2=model2.predict(X_new1)

#Metrics
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score
cm2=confusion_matrix(Y,Y_pred2)
acc2=accuracy_score(Y,Y_pred2)
rec2=recall_score(Y,Y_pred2)
f12=f1_score(Y,Y_pred2)

print('Confusion matrix:',cm2)
print('Accuracy Score:',(acc2*100).round(3))
print('Recall Score:',(rec2*100).round(3))
print('F1 Score:',(f12*100).round(3))

'''
#Inference: Overall accuracy of the model is achievedd upto 89%
#Best model fit is by considering only the numerical X variables with accuracy of 88%
'''




© 2022 GitHub, Inc.
Terms
Privacy
Security
Status
Docs
Contact GitHub
Pricing
API
Training
Blog
About
Loading complete