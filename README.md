import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# 1. load dataset

df = pd.read_csv('Train.csv')

# 2. check null values

df.isnull().sum()

# 3. print information about dataset

df.info()

# 4. Describe dataset in statistic form

df.describe()

df.head()

# 5. drop underrisk column

df_1 = df.drop('UnderRisk',axis=1)

# 1. Apply Kmean clustering on dataset 

from sklearn.cluster import KMeans


cls = KMeans(n_clusters = 2, )
cls.fit(df_1)

# 2. check the wcss score

wcss = cls.inertia_
wcss

# 3 . try different n and find wcss score

#create empty list
wcss = []
#select k value from 1 to 10
for i in range(1, 11):
    cls = KMeans(n_clusters = i, random_state = 42)
    cls.fit(df_1)
    # inertia method returns wcss for that model
    wcss.append(cls.inertia_)

# 4. plot all wcss score

plt.figure(figsize=(10,5))
sns.lineplot(range(1, 11), wcss,marker='o',color='red')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# 5 Try again kmeans with best no. cluster according wo wcss score

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 6, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(df_1)



y_kmeans

# 6. print cluster centers 

kmeans.cluster_centers_

# 7. create column cluster for predicted labels value

df_1['cluster']=  y_kmeans
df_1.head()

# 8. Plot the hierarchical clustering using scipy 

df_h = df.drop('UnderRisk',axis=1)

#The following linkage methods are used to compute the distance between two clusters 
# method='ward' uses the Ward variance minimization algorithm
from scipy.cluster.hierarchy import linkage,dendrogram
merg = linkage(df_1, method = "ward")
#Plot the hierarchical clustering as a dendrogram.
#leaf_rotation : double, optional Specifies the angle (in degrees) to rotate the leaf labels.

dendrogram(merg, leaf_rotation = 90)
plt.xlabel("data points")
plt.ylabel("euclidean distance")
plt.show()

# 9. Apply AgglomerativeClustering using number of cluster

from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters = 6, affinity = "euclidean", linkage = "ward")
cluster = hc.fit_predict(df_h)

# 10. create label column for predicted cluster label

df_h["label"] = cluster

df_h.head()

# 11 .show label counts 

df_h.label.value_counts()


# 12 . show a silhouette score

from sklearn.metrics import silhouette_score

score_agg = silhouette_score(df_h, cluster)
score_agg

