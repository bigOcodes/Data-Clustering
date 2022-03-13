import numpy as np
import pandas as pd
import matplotlib.pyplot as mpt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering as ac

csv_file = pd.read_csv("data2.csv")
data = csv_file.iloc[:,0:].values

# dendro = sch.dendrogram(sch.linkage(data, method='ward'))
# mpt.title('Dendrogram plot')
# mpt.show()

hc = ac(n_clusters=6,affinity='euclidean',linkage='ward')
y_pred = hc.fit_predict(data)

mpt.scatter(data[y_pred == 0,0],data[y_pred==0,1],s = 40,c = 'blue',label='Cluster1')
mpt.scatter(data[y_pred == 1,0],data[y_pred==1,1],s = 40,c = 'red',label='Cluster2')
mpt.scatter(data[y_pred == 2,0],data[y_pred==2,1],s = 40,c = 'magenta',label='Cluster3')
mpt.scatter(data[y_pred == 3,0],data[y_pred==3,1],s = 40,c = 'cyan',label='Cluster4')
mpt.scatter(data[y_pred == 4,0],data[y_pred==4,1],s = 40,c = 'yellow',label='Cluster5')
mpt.scatter(data[y_pred == 5,0],data[y_pred==5,1],s = 40,c = 'green',label='Cluster6')
mpt.legend()
mpt.show()