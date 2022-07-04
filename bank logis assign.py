# import pandas
import pandas as pd
df = pd.read_csv("C://Users/NAVEEN REDDY/Downloads/bank-full.csv",sep = ';')
df.shape
list(df)
type(df)
df.dtypes
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df["job_code"] = LE.fit_transform(df["job"])
df[["job", "job_code"]].head(17)
pd.crosstab(df.job,df.job_code)

df["marital_code"] = LE.fit_transform(df["marital"])
df[["marital", "marital_code"]].head(17)
pd.crosstab(df.marital,df.marital_code)

df["education_code"] = LE.fit_transform(df["education"])
df[["education", "education_code"]].head(17)
pd.crosstab(df.education,df.education_code)

df["default_code"] = LE.fit_transform(df["default"])
df[["default", "default_code"]].head(17)
pd.crosstab(df.default,df.default_code)

df["housing_code"] = LE.fit_transform(df["housing"])
df[["housing", "housing_code"]].head(17)
pd.crosstab(df.housing,df.housing_code)

df["loan_code"] = LE.fit_transform(df["loan"])
df[["loan", "loan_code"]].head(17)
pd.crosstab(df.loan,df.loan_code)

df["contact_code"] = LE.fit_transform(df["contact"])
df[["contact", "contact_code"]].head(17)
pd.crosstab(df.contact,df.contact_code)

df["month_code"] = LE.fit_transform(df["month"])
df[["month", "month_code"]].head(17)
pd.crosstab(df.month,df.month_code)

df["poutcome_code"] = LE.fit_transform(df["poutcome"])
df[["poutcome", "poutcome_code"]].head(17)
pd.crosstab(df.poutcome,df.poutcome_code)

df["y_code"] = LE.fit_transform(df["y"])
df[["y", "y_code"]].head(17)
pd.crosstab(df.y,df.y_code)
df.drop(['job','marital','education','default','housing','loan','contact','month','poutcome','y'],axis=1,inplace=True)
X=df.iloc[:,0:16]
list(X)
Y=df.y_code
# import the class
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
# fit the model with data
logreg.fit(X,Y)
logreg.intercept_  ## To check the Bo values
logreg.coef_       ## To check the coefficients (B1,B2,...B8)
#
Y_Pred=logreg.predict(X)
# comparision
from sklearn.metrics import confusion_matrix, accuracy_score,recall_score,precision_score, f1_score
CM = confusion_matrix(Y, Y_Pred)
CM

# Manual calculations
TN = CM[0,0]
FN = CM[1,0]
FP = CM[0,1]
TP = CM[1,1]

# sklearn calculations
print("Accuracy_score:",(accuracy_score(Y,Y_Pred)*100).round(3))
print("Recall/Sensitivity score:",(recall_score(Y,Y_Pred)*100).round(3))
print("Precision score:",(precision_score(Y,Y_Pred)*100).round(3))

Specificity = TN /(TN + FP)
print("Specificity score: ",(Specificity*100).round(3))
print("F1 score: ",(f1_score(Y,Y_Pred)*100).round(3))

# Show confusion matrix in a separate window
import matplotlib.pyplot as plt
plt.matshow(CM)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

from sklearn.metrics import roc_curve,roc_auc_score
logreg.predict_proba(X).shape

logreg.predict_proba(X)[:,1]

print(logreg.predict_proba(X))
y_pred_proba = logreg.predict_proba(X)[:,1]
fpr, tpr,_ = roc_curve(Y,  y_pred_proba)

plt.plot(fpr,tpr)
#plt.legend(loc=4)
plt.ylabel('tpr - True Positive Rate')
plt.xlabel('fpr - False Positive Rate')
plt.show()

# auc scores
auc = roc_auc_score(Y, y_pred_proba)
auc
#1 - Confusion Matrix
#The result is telling us that we have 39455+456 correct predictions and 4833+467 incorrect predictions.
#Accuracy_score: 88.762
#Of the entire data set, 88% of the clients will subscribe
