# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 09:08:02 2020

@author: Benjamim
"""

from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


dados = pd.read_csv('filtered_dataset', sep= ';')
'''
1-cura
2-obito
3-obito outras causas
9-ignorado
''' 
clss = dados[['EVOLUCAO','NU_IDADE_N']].fillna(0)
''' .fillna(0) troca valores Nan por 0'''
kmeans = KMeans(n_clusters=5,verbose=5)
kmeans.fit(clss)
y_kmeans = kmeans.predict(clss)

'''criando vizualização dos centroides'''
from sklearn.metrics import pairwise_distances_argmin
def find_clusters(clss, n_clusters, rseed=2):
    # 1. Randomly choose clusters
    rng = np.random.RandomState(rseed)
    i = rng.permutation(clss.shape[0])[:n_clusters]
    centers = clss[i]
    
    centers_his = []
    labels_his = []
    
    while(True):
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(clss, centers)
        
        # 2b. Find new centers from means of points
        new_centers = np.array([clss[labels == i].mean(0)
                                for i in range(n_clusters)])
        
        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers
        
        centers_his.append(centers)
        labels_his.append(labels)
    
    return centers, labels, centers_his, labels_his

centers, labels, centers_his, labels_his = find_clusters(clss, 5)

for i in range(len(centers_his)):
    centers = centers_his[i]
    labels = labels_his[i]
    fig = plt.figure(figsize=(7, 5))
    fig.set_tight_layout(True)
    plt.scatter(clss[:, 0], clss[:, 1], c=labels,
                s=50, cmap='rainbow');
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
    plt.savefig('kmeans_demo/{}.png'.format(i))

