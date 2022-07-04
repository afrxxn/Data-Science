import pandas as pd
data=pd.read_csv("C://Users/NAVEEN REDDY/Downloads/Cutlets.csv")
data.shape
list(data)

'''
#Test of Hypothesis
Ho: UnitA = UnitB ---> No significant difference in diameter of cutlets of two units
H1: UnitA != UnitB ---> Significant difference in diameter of cutlets of two units
'''
uA=data['Unit A']
uB=data['Unit B']
alpha=0.05 #alpha is the level of significance
from scipy.stats import ttest_ind
z,p=ttest_ind(uA,uB)
print(z,p)

if p>alpha:
    print('Accept Ho and Reject H1')
else:
    print('Accept H1 and Reject Ho')
    
#Inference: As we have got to accept Ho that implies there is No significant
