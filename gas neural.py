#import pandas
import numpy as np
import pandas as pd 
df=pd.read_csv("C://Users/NAVEEN REDDY/Downloads/gas_turbines.csv")
df.head()
df.shape
df.isna().sum()
list(df)
df.drop([ 'AFDP', 'GTEP','TAT','CO', 'NOX','TIT','CDP'],axis=1,inplace=True)
x=df.iloc[:,0:3]
y=df["TEY"]


from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.models import Sequential
import keras

model = Sequential()
model.add(Dense(5, input_dim=3,  activation='relu'))
model.add(Dense(1, activation='relu')) 
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x, y, validation_split=0.25, epochs=50, batch_size=10)
scores = model.evaluate(x, y)

y_pred=model.predict(x)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y,y_pred)
mse

