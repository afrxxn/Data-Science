import pandas as pd
df = pd.read_csv("C://Users/NAVEEN REDDY/Downloads/LabTAT.csv")
df.shape
list(df)

df['Laboratory 1'].hist()
df['Laboratory 2'].hist()
df['Laboratory 3'].hist()
df['Laboratory 4'].hist()
#Test of Hypothesis
''' Ho: l1 = l2 = l3 = l4 ---> All laboratories avg TAT is same
 H1: l1 != l2 != l3 != l4 ---> Any one of the laboratories avg TAT among the 4 are not same
'''
l1=df['Laboratory 1']
l2=df['Laboratory 2']
l3=df['Laboratory 3']
l4=df['Laboratory 4']
import scipy.stats as stats
z,p=stats.f_oneway(l1,l2,l3,l4)
print(z,p)


alpha = 0.05 

if p< alpha:
    print("Ho is rejected and H1 is accepted")
else:
    print("H1 is rejected and H0 is accepted")
    

#conclusion
# thus H1 is accepted
# Variance of all 4 laboratories are the not same   
    
    
    