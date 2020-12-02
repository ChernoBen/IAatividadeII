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
'''
1-cura
2-obito
3-obito outras causas
9-ignorado
''' 

dados = pd.read_csv('filtered_dataset', sep= ';')

''' .fillna(0) troca valores Nan por 0'''
clss = dados[['NU_IDADE_N','EVOLUCAO']].fillna(0)
'''removendo tudo que nao seja cura e obito'''
clss = clss.drop(clss[clss['EVOLUCAO'] > 2 ].index)
'''removendo valores nao informados'''
clss = clss.drop(clss[clss['EVOLUCAO'] < 1 ].index)

classe = clss.values
'''organizando kmeans'''
cluster = KMeans(n_clusters = 2)
cluster.fit(classe)
y_kmeans = cluster.predict(classe)

#visualizando centroides
centroides = cluster.cluster_centers_
centroides

#visualização dos grupos que cada reistro foi associado
previsoes = cluster.labels_
previsoes 

#contagem dos registros por classe
unicos2, quantidade2 = np.unique(previsoes,return_counts = True)
quantidade2

#geração do grafico com os clusters gerados, considerando para um (previsoes 0,1 ou 2)
#Usamos somente as colunas 0 e 1 da base de dados original para termos 2 dimensoes
'''plt.scatter(clss[previsoes == 0, 0],clss[previsoes == 0, 1],
            c = 'green',label = 'obtios')
plt.scatter(clss[previsoes == 1, 0],clss[previsoes == 1, 1],
            c = 'red',label = 'cura')

plt.legend()''' 
#tentativa plot

from sklearn.metrics import pairwise_distances_argmin

def find_clusters(X, n_clusters, rseed=2):
    # 1. Randomly choose clusters
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    
    centers_his = []
    labels_his = []
    
    while(True):
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(X, centers)
        
        # 2b. Find new centers from means of points
        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(n_clusters)])
        
        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers
        
        centers_his.append(centers)
        labels_his.append(labels)
    
    return centers, labels, centers_his, labels_his


   
centers, labels, centers_his, labels_his = find_clusters(classe, 2)
#plot centroides
for i in range(len(centers_his)):
    centers = centers_his[i]
    labels = labels_his[i]
    fig = plt.figure(figsize=(7, 5))
    fig.set_tight_layout(True)
    plt.scatter(classe[:, 0], classe[:, 1], c=labels,label = 'Obitos',
                s=50, cmap='rainbow');
    plt.scatter(centers[:, 0], centers[:, 1], c='black', label='Curados',s=200, alpha=0.5);
    #plt.savefig('{}.png'.format(i))




