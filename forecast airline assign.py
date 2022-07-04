#import pandas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
airline = pd.read_excel("C://Users/NAVEEN REDDY/Downloads/Airlines+Data.xlsx")
airline.shape
list(airline)
airline.Passengers.plot()#plot 
airline.Month.plot()
# heat map
plt.figure(figsize=(12,8))
heatmap_y_month = pd.pivot_table(data=airline,values="Passengers",index="year",columns="month",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g") 
# t-value
t=[]
for i in range(1,97):
    t.append(i)
t
# finding log_psg,t,t_sq
airline['t'] = pd.DataFrame(t)    
airline['log_psg'] = np.log2(airline['Passengers'])
airline['t_sq'] = airline['t']*airline['t']
airline
# spliting  date,year,month 
airline["Date"] = pd.to_datetime(airline.Month,format="%b-%y")
airline["month"] = airline.Date.dt.strftime("%b") 
airline["year"] = airline.Date.dt.strftime("%Y") 
airline1 = airline.copy()
# creating dummies for moonth
airline1 = pd.get_dummies(airline, columns = ['month'])
airline1
# boxplot
plt.figure(figsize=(8,6))
sns.boxplot(x="Month",y="Passengers",data=airline)
# spliting the data to train and test
airline1.shape
Train = airline1.head(85)
Test = airline1.tail(11)
Test
# fitting the model by using train ,test and t,t_sq,log_psg,dummies
import statsmodels.formula.api as smf 

#Linear Model

linear_model = smf.ols('Passengers~t',data=Train).fit()
pred_linear = pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_linear))**2))
rmse_linear

#Exponential
Exp = smf.ols('log_psg~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp

#Quadratic 
Quad = smf.ols('Passengers~t+t_sq',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_sq"]]))
#pred_Quad = pd.Series(Exp.predict(pd.DataFrame(Test[["t","t_square"]))) # we hve to verify
rmse_Quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_Quad))**2))
rmse_Quad

#Additive seasonality 
add_sea = smf.ols('Passengers~month_Jan+month_Feb+month_Mar+month_Apr+month_May+month_Jun+month_Jul+month_Aug+month_Sep+month_Oct+month_Nov+month_Dec',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['month_Jan','month_Feb','month_Mar','month_Apr','month_May','month_Jun','month_Jul','month_Aug','month_Sep','month_Oct','month_Nov','month_Dec']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea))**2))
rmse_add_sea

#Additive Seasonality Quadratic 
add_sea_Quad = smf.ols('Passengers~t+t_sq+month_Jan+month_Feb+month_Mar+month_Apr+month_May+month_Jun+month_Jul+month_Aug+month_Sep+month_Oct+month_Nov+month_Dec',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['month_Jan','month_Feb','month_Mar','month_Apr','month_May','month_Jun','month_Jul','month_Aug','month_Sep','month_Oct','month_Nov','month_Dec','t','t_sq']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad

##Multiplicative Seasonality
Mul_sea = smf.ols('log_psg~month_Jan+month_Feb+month_Mar+month_Apr+month_May+month_Jun+month_Jul+month_Aug+month_Sep+month_Oct+month_Nov+month_Dec',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea


#Multiplicative Additive Seasonality 
Mul_Add_sea = smf.ols('log_psg~t+month_Jan+month_Feb+month_Mar+month_Apr+month_May+month_Jun+month_Jul+month_Aug+month_Sep+month_Oct+month_Nov+month_Dec',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 

#Compare the results 
data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
type(data)

table_rmse=pd.DataFrame(data)
table_rmse.sort_values(['RMSE_Values'])

''' inference: after fitting the model rmse_add_sea_quad and rmse_quad is better
 and 12 dummies created by month'''