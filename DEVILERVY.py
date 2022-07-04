import numpy as np
import pandas as pd

df = pd.read_csv("C://Users/NAVEEN REDDY/Downloads/delivery_time (2).csv")
df.shape
type(df)
list(df)
df.ndim

X = df['Delivery Time'] 
X.shape
X.ndim
type(X)

X = X[:, np.newaxis] 
X.ndim
type(X)

Y = df[ 'Sorting Time']  
Y.shape
Y.ndim


df.plot.scatter(x='Delivery Time', y= 'Sorting Time')
df.corr()


from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X, Y)
model.intercept_  
model.coef_



Y_Pred = model.predict(X)

# Plot outputs
import matplotlib.pyplot as plt
plt.scatter(X, Y,  color='black')
plt.plot(X, Y_Pred, color='red')
plt.show()


Y_error = Y-Y_Pred
print(Y_error)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_Pred)
mse

RMSE = np.sqrt(mse)
from sklearn.metrics import r2_score
Rsquare = r2_score(Y,Y_Pred)
print(Rsquare.round(3))