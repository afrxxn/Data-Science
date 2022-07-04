
import pandas as pd  
import numpy as np  

df = pd.read_csv('C://Users/NAVEEN REDDY/Downloads/crime_data.csv', delimiter=',') 
df.shape
df.head()
X = df.iloc[:, 1:5].values 
X.shape
list(X)
import scipy.cluster.hierarchy as shc

# construction of Dendogram
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 7))  
plt.title("X Dendograms")  
dend = shc.dendrogram(shc.linkage(X, method='complete')) 

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete')
Y = cluster.fit_predict(X)

plt.figure(figsize=(10, 7))  
plt.scatter(X[:,0], X[:,1], c=cluster.labels_, cmap='rainbow')

Y_clust = pd.DataFrame(Y)
Y_clust[0].value_counts()

%matplotlib qt
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2])
plt.show()

from sklearn.cluster import KMeans
KMeans()
kmeans = KMeans(n_clusters=4)

kmeans = kmeans.fit(X)

labels = kmeans.predict(X)
type(labels)

C = kmeans.cluster_centers_
 
kmeans.inertia_

%matplotlib qt
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2])  
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c='Red', s=1000)  

Y = pd.DataFrame(labels)
X  

df_new = pd.concat([pd.DataFrame(X),Y],axis=1)

pd.crosstab(Y[0],Y[0])

Y
clust = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i,random_state=0)
    kmeans.fit(X)
    clust.append(kmeans.inertia_)
    
plt.plot(range(1, 11), clust)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('inertial values')
plt.show()

print(clust)
df.drop(['Unnamed: 0'],axis=1,inplace=True)
array=df.values
array

# DBSCAN 
from sklearn.preprocessing import StandardScaler
stscaler = StandardScaler().fit(array)
X = stscaler.transform(array)
X


from sklearn.cluster import DBSCAN
DBSCAN()
dbscan = DBSCAN(eps=2, min_samples=6)
dbscan.fit(X)

#Noisy samples are given the label -1.
dbscan.labels_

cl=pd.DataFrame(dbscan.labels_,columns=['cluster'])
cl
cl['cluster'].value_counts()


clustered = pd.concat([df,cl],axis=1)

noisedata = clustered[clustered['cluster']==-1]
finaldata = clustered[clustered['cluster']==0]

clustered

a=0
while a<5:
  print(a)
  a=a+1
clustered.mean()
finaldata.mean()
''' conclusion :As with murder and assault, also rates of rape and assault show clearer correlations
Analysing Murder and Assault variables shows a clearer connection between them.
 Higher the murder rates in a city higer the assaults '''