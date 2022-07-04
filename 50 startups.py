#import pandas 
import pandas as pd
df = pd.read_csv("C://Users/Downloads/50_Startups.csv")
df.shape
list(df)
df.dtypes
df.head()
df
import matplotlib.pyplot as plt
df.plot.scatter(x='R&D Spend',y='Profit')
df.plot.scatter(x='State',y='Profit')            # scatter for all x-variables and y -variables
df.plot.scatter(x= 'Administration',y='Profit')
df.plot.scatter(x= 'Marketing Spend',y='Profit')
import seaborn as sns
corr = df.corr()
sns.heatmap(corr, annot=True, square=True)    # heat map to check correct correlation
plt.yticks(rotation=3)
plt.show()

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()                              # state is in categorical so we do one heart or label endcoding 
df['State']= LE.fit_transform(df['State'])
pd.crosstab(df['State'],df['State'])
df.head()

Y = df['Profit']
X = df[['Marketing Spend','R&D Spend','Administration']]
X = df[['R&D Spend','Administration']]
# Import Linear Regression
from sklearn.linear_model import LinearRegression 
model = LinearRegression().fit(X, Y)
Y_Pred = model.predict(X)

import statsmodels.api as sm
X1 = sm.add_constant(X)      ## let's add an intercept (beta_0) to our model
from statsmodels.stats.outliers_influence import variance_inflation_factor   # for multi collinearity and numerical problems we use VIF method
vif = [variance_inflation_factor(X1.values, j) for j in range(X1.shape[1])]
variable_VIF = pd.concat([pd.DataFrame(X1.columns),pd.DataFrame(np.transpose(vif))], axis = 1)
print(variable_VIF)

from sklearn.metrics import r2_score
r2 = r2_score(Y,Y_Pred)*100  
print("R square: ", r2.round(3))

import numpy as np
n = X.shape[0]
k = X.shape[1] + 1
ssres = np.sum((Y - Y_Pred)**2  )
sstot = np.sum((Y - np.mean(Y))**2  )
num = ssres/(n-k)
den = sstot/(n-1)
r2_adj =  1  - (num/den)
print("Adjusted Rsquare: ", (r2_adj*100).round(3))

''' conclusion: R sq and p Value of the Model is Good and the model can be accepted. 
However as you can see not all variables have acceptable p value.'''




