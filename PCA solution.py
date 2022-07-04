# Load the data
import pandas as pd
df = pd.read_csv("C://Users/NAVEEN REDDY/Downloads/wine.csv")
df.shape
df
list(df)

df['Type'].value_counts()

# Recode the Y variable
def f1(x):
    if x == 1:
        return 0
    elif x == 2:
        return 1
    else:
        return 2

df['Type_new'] = df['Type'].apply(f1)
df['Type_new'].value_counts()

x1 = df.iloc[:,1:14]
x1
list(x1)
x1.shape

# Standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x1_scale = scaler.fit_transform(x1)


#=================================================================================

# principal component analysis
# load decomposition to do PCA analysis with sklearn
from sklearn.decomposition import PCA
PCA()
pca = PCA(n_components=3)
pc = pca.fit_transform(x1_scale)
pc.shape

pc_df = pd.DataFrame(data = pc , columns = ['PC1', 'PC2','PC3'])
pc_df.head()

pca.explained_variance_ratio_
sum(pca.explained_variance_ratio_)

import seaborn as sns
df_l1 = pd.DataFrame({'var':pca.explained_variance_ratio_,'PC':['PC1','PC2','PC3']})
sns.barplot(x='PC',y="var", data=df_l1, color="c")

X = pc_df.iloc[:,:3]
X


#=================================================================================

# K means clustering
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X.iloc[:, 0], X.iloc[:, 1], X.iloc[:, 2])
plt.show()

# intializing Kmeans
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3,random_state=4)
# fitting with inputs
km.fit(X)
# predicting the clusters
labels = km.predict(X)
type(labels)
# Getting the clusters
C = km.cluster_centers_
# total with in centroid sum of squares
km.inertia_

Y_knn = pd.DataFrame(labels)
X
df_new = pd.concat([pd.DataFrame(X),Y_knn],axis=1)

df_new.rename(columns={0:'Knn_Y'},inplace=True)
list(df_new)

# Elbow plot
clust = []
for i in range(1,11):
    km = KMeans(n_clusters=i,random_state=4)
    km.fit(X)
    clust.append(km.inertia_)
    
plt.plot(range(1,11),clust) 
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('inertial values') 
plt.show()    

print(clust)

# Confusion Matrix for accuracy score
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(df['Type_new'],df_new['Knn_Y'])
cm

ac = accuracy_score(df['Type_new'],df_new['Knn_Y'])
ac
acscore = (ac*100).round(3)
print("Accuracy score:",acscore)

#=================================================================================

# Agglomerative clustering

import scipy.cluster.hierarchy as shc

# construction of dendogram
import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))
plt.title("customer dendograms")
dend = shc.dendrogram(shc.linkage(X , method='complete'))
plt.show()

# forming a group usin clusters
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=3,linkage='complete')
y_hc = cluster.fit_predict(X)

Y_hc = pd.DataFrame(y_hc)
Y_hc[0].value_counts()
df['Type_new'].value_counts()

# Confusion Matrix for accuracy score
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(df['Type_new'],Y_hc[0])
cm
ac = accuracy_score(df['Type_new'],Y_hc[0])
ac

'''
Inference:
        k means clustering  gives best accuracy score :96%
       k means clustering is better than hierarichal clustering 
'''