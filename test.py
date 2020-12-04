# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 19:54:59 2020

@author: Benjamim
"""


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

dados = pd.read_csv('https://raw.githubusercontent.com/ChernoBen/IAatividadeII/main/new_dataset', sep= ';')
teste = dados
#primeiro metodo para plot/df tipo int
teste2 = teste.values
'''removendo valores diferentes de obitos e cura'''
dados  = dados.drop(dados[dados['EVOLUCAO'] > 2 ].index)
dados  = dados.drop(dados[dados['EVOLUCAO'] < 1  ].index)
   
'''tranformando valores em rotulos'''
#tst = teste['EVOLUCAO'].apply(preprocessing.LabelEncoder().fit_transform)        
teste = dados.apply(preprocessing.LabelEncoder().fit_transform)

# visualização de quantos registros existem por classe
unicos,quantidade = np.unique(teste,return_counts=True)
teste3 = teste.to_numpy()

'''talvez idade por outro fator'''
teste4 = teste[['FATOR_RISC','NU_IDADE_N']].values

#instanciando KMeans/ criando agrupamentos
cluster = KMeans(n_clusters=2)
cluster.fit(teste4)
pred = cluster.predict(teste4)

#visualização dos centroides(agrupamentos ou clusters anteiormente definidos)
centroides = cluster.cluster_centers_

#visualização dos grupos que cada registro foi associado
previsoes = cluster.labels_

#contagem dos registros por classe
unicos2, quantidade2 = np.unique(previsoes,return_counts = True)

#geração da matriz de contingencia para comparar os grupos com a base de dados
resultados = confusion_matrix(teste4[:,0],previsoes)
 
'''-----------------------''' 
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
        '''verifica se todos os elementos do array são True e retorna True'''
        if np.all(centers == new_centers):
            break
        centers = new_centers
        centers_his.append(centers)
        labels_his.append(labels)
    return centers, labels, centers_his, labels_his

'''teste3 tem que receber apenas 2 colunas'''
centers, labels, centers_his, labels_his = find_clusters(teste4, 2)
'''-------------------------'''


for i in range(len(centers_his)):
    #talves usar os centroides delcarados inicialmente
    centers = centers_his[i]
    labels = labels_his[i]
    fig = plt.figure(figsize=(7, 5))
    fig.set_tight_layout(True)
    plt.scatter(teste4[:, 0], teste4[:, 1], c=labels,
                s=50, cmap='rainbow');
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
    plt.savefig('kmeans_demo/{}.png'.format(i))








