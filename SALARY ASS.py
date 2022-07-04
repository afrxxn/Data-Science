#import pandas
import numpy as np
import pandas as pd
df = pd.read_csv("C://Users/NAVEEN REDDY/Downloads/Salary_Data.csv")
df.shape
type(df)
list(df)
df.ndim
X = df['YearsExperience'] 
X.shape
type(X)
Y = df['Salary']  
Y.shape
Y.ndim
df.plot.scatter(x='YearsExperience', y='Salary') # as showing the scatter plot of X and Y variables
df.corr()
X = X[:, np.newaxis] 
X.ndim
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X, Y)             # fitting model X and Y
Y_Pred = model.predict(X)

# Plot outputs
import matplotlib.pyplot as plt

Y_error = Y-Y_Pred
print(Y_error)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_Pred)
mse
RMSE = np.sqrt(mse)
RMSE
from sklearn.metrics import r2_score
Rsquare = r2_score(Y,Y_Pred)
print(Rsquare.round(3))
'''  conclusion :
    As per r sq and mse ,mse is value is high so we have to use transformations like square and log transformations '''

#Applying Transformations on X variable
#Exploratory Data Analysis
df['Sq YE']=np.sqrt(df['YearsExperience']) #Square root transformation on X
X1=df['Sq YE']
X1=X1[:,np.newaxis]
X1.ndim
import seaborn as sns
sns.distplot(X1)
df['Sq YE'].skew() #Skewness is 0.379, it can be accpeted as it is under range of -0.5 to +0.5
df['Sq YE'].describe()

df['lg Salary']=np.log(df['Salary']) #Log transformation on Y
Y1=df['lg Salary']
sns.distplot(Y1)

['lg Salary'].skew()
df['lg Salary'].describe()
#Scatter plot
df.plot.scatter(x='Sq YE',y='lg Salary')
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
''' conclusion:  
    After applying transformation mean square error reduces the data is good '''
