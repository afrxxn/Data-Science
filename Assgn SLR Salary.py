#Simple linear regression to predict salary
import pandas as pd
import numpy as np

data=pd.read_csv("F:\\ExcelR\\Assignments\\Simple Linear regression\\Salary_Data.csv")
data.shape
list(data)
data.corr()

#Exploratory Data Analysis
import seaborn as sns
X=data['YearsExperience']
X=X[:,np.newaxis]
X.ndim
Y=data['Salary']
sns.distplot(data['YearsExperience'])
data['YearsExperience'].hist() #Here as per the histogram, it looks like positively skewed
data['YearsExperience'].skew() #Skewness is 0.379, it can be accpeted as it is under range of -0.5 to +0.5
data['YearsExperience'].describe()

sns.distplot(data['Salary'])
data['Salary'].hist()
data['Salary'].skew()
data['Salary'].describe()

#Scatter plot
data.plot.scatter(x='YearsExperience',y='Salary')

#Fitting the model
from sklearn.linear_model import LinearRegression
model=LinearRegression().fit(X,Y)
model.intercept_
model.coef_
Y_pred=model.predict(X)

#To draw plots
import matplotlib.pyplot as plt
plt.scatter(X,Y,color='Blue')
plt.plot(X,Y_pred,color='Red')
plt.show()

#Finding error
Y_error=Y-Y_pred
sns.distplot(Y_error) #The errors are following normal distribution.

#Metrics
from sklearn.metrics import mean_squared_error
MSE=mean_squared_error(Y,Y_pred)
print(MSE)
from math import sqrt
RMSE=np.sqrt(MSE)
print(RMSE)
'''
Inference: The Mean Square Error and Root Mean Square Error are quite high
           So, trying to apply transformations on X and Y in order to reduce error

'''
#################################################################################################

#Applying Transformations on X variable
#Exploratory Data Analysis
data['Sq YE']=np.sqrt(data['YearsExperience']) #Square root transformation on X
X1=data['Sq YE']
X1=X1[:,np.newaxis]
X1.ndim
sns.distplot(X1)
data['Sq YE'].skew() #Skewness is 0.379, it can be accpeted as it is under range of -0.5 to +0.5
data['Sq YE'].describe()

data['lg Salary']=np.log(data['Salary']) #Log transformation on Y
Y1=data['lg Salary']
sns.distplot(Y1)
data['lg Salary'].skew()
data['lg Salary'].describe()

#Scatter plot
data.plot.scatter(x='Sq YE',y='lg Salary')

#Fitting the model
from sklearn.linear_model import LinearRegression
model1=LinearRegression().fit(X1,Y1)
model1.intercept_
model1.coef_
Y_pred1=model1.predict(X1)

sns.regplot(x=X1,y=Y1,color='Blue')

#Finding error
Y_error1=Y_pred1-Y1
sns.distplot(Y_error1) #The errors looks to follow normal distribution

#Plot
import matplotlib.pyplot as plt
plt.scatter(X1,Y1,color='Blue')
plt.plot(X1,Y_pred1,color='Red')
plt.show()

#Metrics
from sklearn.metrics import mean_squared_error
MSE1=mean_squared_error(Y1,Y_pred1)
print(MSE1)
from math import sqrt
RMSE1=np.sqrt(MSE1)
print(RMSE1)

#Fitting the using statsmodels package
import statsmodels.api as sma
model2=sma.OLS(X1,Y1).fit()
Y_pred2=model2.predict(X1)
model2.summary()
'''
Inference: The Mean Square Error and Root Mean Square Error after applying transformations
            are getting reduced by large amounts and errors are also following the normal distribution
            even after applying transformations, seems to be a good model for the given dataset.
'''