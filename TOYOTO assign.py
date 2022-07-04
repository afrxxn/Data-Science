import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("C://Users/NAVEEN REDDY/Downloads/ToyotaCorolla.csv",encoding='latin1')
df.shape
list(df)
df.dtypes
df.head()
df
import matplotlib.pyplot as plt
df.plot.scatter(x='Age_08_04',y="Price")
df.plot.scatter(x='KM',y="Price")            # scatter for all x-variables and y -variables
df.plot.scatter(x= 'HP',y="Price")
df.plot.scatter(x= 'cc',y="Price")
df.plot.scatter(x= 'Doors',y="Price")
df.plot.scatter(x= 'Gears',y="Price")
df.plot.scatter(x= 'Quarterly_Tax',y="Price")
df.plot.scatter(x= 'Weight',y="Price")
y=df.Price
x=df.iloc[:,1:9]
df.drop(['Id','Model','Mfg_Month','Mfg_Year','Fuel_Type','Met_Color','Color','Automatic','Cylinders',
         'Mfr_Guarantee','BOVAG_Guarantee','Guarantee_Period','ABS','Airbag_1','Airbag_2',
         'Airco','Automatic_airco','Boardcomputer','CD_Player','Central_Lock','Powered_Windows','Power_Steering',
         'Radio','Mistlamps','Sport_Model','Backseat_Divider','Metallic_Rim','Radio_cassette','Tow_Bar'],axis=1,inplace=True)
import seaborn as sns
corr = df.corr()
sns.heatmap(corr, annot=True, square=True)
plt.yticks(rotation=3)
plt.show()

Y = df['Price']
#X = df[['KM','Age_08_04','HP','cc','Doors','Gears','Quarterly_Tax','Weight']]
#X = df[['KM','Age_08_04','HP','Quarterly_Tax','Weight']]
X = df[['KM','Age_08_04','HP','Weight']]  # best model 
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(x, y)
model.intercept_ 
model.coef_       
Y_Pred = model.predict(x)

import statsmodels.api as sm
x1 = sm.add_constant(x) ## let's add an intercept (beta_0) to our model
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = [variance_inflation_factor(x1.values, j) for j in range(x1.shape[1])]
variable_VIF = pd.concat([pd.DataFrame(x1.columns),pd.DataFrame(np.transpose(vif))], axis = 1)
print(variable_VIF)

from sklearn.metrics import r2_score
r2 = r2_score(y,Y_Pred)*100  
print("R square: ", r2.round(3))

import statsmodels.api as sma
x1 = sma.add_constant(x)
lm2 = sma.OLS(y,x1).fit()
lm2.summary()
''' conclusion: R sq and p Value of the Model is Good and the model can be accepted. 
However as you can see not all variables have acceptable p value.'''


