import pandas as pd 
import numpy as np
import pandas as pd
data=pd.read_csv("C://Users/NAVEEN REDDY/Downloads/BuyerRatio.csv")
data.shape
list(data)
#a=50+142+131+70=393
#b=435+1523+1356+750=4064
#east
a=393
b=4064

BE1=50/a
BE2=435/b
print(BE1,BE2)

nump=np.array([BE1,BE2])
count=np.array([a,b])
from statsmodels.stats.proportion import proportions_ztest
stat,BE = proportions_ztest(count, nump)

alpha = 0.05
# Ho: male and females are same
# H1: males amd females are not same
if BE < alpha:
    print("Ho is rejected and H1 is accepted")
else:
    print("H1 is rejected and H0 is accepted")

#west
BE1=142/a
BE2=1523/b
print(BE1,BE2)

nump=np.array([BE1,BE2])
count=np.array([a,b])
from statsmodels.stats.proportion import proportions_ztest
stat,BE = proportions_ztest(count, nump)

alpha = 0.05
# Ho: male and females are same
# H1: males amd females are not same
if BE < alpha:
    print("Ho is rejected and H1 is accepted")
else:
    print("H1 is rejected and H0 is accepted")

#north
BE1=131/a
BE2=1356/b
print(BE1,BE2)

nump=np.array([BE1,BE2])
count=np.array([a,b])
from statsmodels.stats.proportion import proportions_ztest
stat,BE = proportions_ztest(count, nump)

alpha = 0.05
# Ho: male and females are same
# H1: males amd females are not same
if BE < alpha:
    print("Ho is rejected and H1 is accepted")
else:
    print("H1 is rejected and H0 is accepted")

#south
BE1=70/a
BE2=750/b
print(BE1,BE2)

nump=np.array([BE1,BE2])
count=np.array([a,b])
from statsmodels.stats.proportion import proportions_ztest
stat,BE = proportions_ztest(count, nump)

alpha = 0.05
# Ho: male and females are same
# H1: males amd females are not same
if BE < alpha:
    print("Ho is rejected and H1 is accepted")
else:
    print("H1 is rejected and H0 is accepted")
    
''' inference:
        hence , males and females are same '''







