import numpy as np
import pandas as pd
import matplotlib.pyplot as mpt
from sklearn.cluster import KMeans

csv_file = pd.read_csv('dataset_file.csv')
data = csv_file.iloc[:,0:].values

kmeans = KMeans(n_clusters=2,init='k-means++',random_state=1)
y_pred = kmeans.fit_predict(data)

mpt.scatter(data[y_pred==0,0],data[y_pred==0,1],s=40,c='cyan',label='cluster1')
mpt.scatter(data[y_pred==1,0],data[y_pred==1,1],s=40,c='blue',label='cluster2')
mpt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='black',label='centroid')
mpt.title('Clusters of customers')
mpt.legend()
mpt.show()