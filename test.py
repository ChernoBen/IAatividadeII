# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 19:54:59 2020

@author: Benjamim
"""

from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing

'''
1-cura
2-obito
3-obito outras causas
9-ignorado
remover casos de item 3 e 9

tranformar cada instancia de cada coluna em labels 

talvez remover coluna grupo de risco 
''' 

dados = pd.read_csv('new_dataset', sep= ';')
teste = dados

'''removendo valores diferentes de obitos e cura'''
teste  = teste.drop(teste[teste['EVOLUCAO'] > 2 ].index)
teste  = teste.drop(teste[teste['EVOLUCAO'] < 1  ].index)

'''tranformando valores em rotulos'''
#tst = teste['EVOLUCAO'].apply(preprocessing.LabelEncoder().fit_transform)        
teste = dados.apply(preprocessing.LabelEncoder().fit_transform)

# visualização de quantos registros existem por classe
unicos,quantidade = np.unique(teste,return_counts=True)

#instanciando KMeans/ criando agrupamentos
cluster = KMeans(n_clusters=4)
cluster.fit(teste)

#visualização dos centroides(agrupamentos ou clusters anteiormente definidos)
centroides = cluster.cluster_centers_

#visualização dos grupos que cada registro foi associado
previsoes = cluster.labels_

#contagem dos registros por classe
unicos2, quantidade2 = np.unique(previsoes,return_counts = True)

#geração da matriz de contingencia para comparar os grupos com a base de dados
resultados = confusion_matrix(teste[:,23],previsoes)

'''--------------------------''' 
#primeiro metodo para plot/df tipo int
teste2 = teste.values

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

'''---------------------------'''
centers, labels, centers_his, labels_his = find_clusters(teste2, 4)

'''-----------------------
#segundo metodo p/plot
#geração do grafico com os clusters gerados, considerando para um (previsoes 0,1 ou 2)
#Usamos somente as colunas 0 e 1 da base de dados original para termos 2 dimensoes
plt.scatter(iris.data[previsoes == 0, 0],iris.data[previsoes == 0, 1],
            c = 'green',label = 'Setosa')
plt.scatter(iris.data[previsoes == 1, 0],iris.data[previsoes == 1, 1],
            c = 'red',label = 'Versicolor')
plt.scatter(iris.data[previsoes == 2, 0], iris.data[previsoes == 2, 1],
            c = 'blue',label = 'Virginica')
plt.legend()
''' 