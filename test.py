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
#primeiro metodo para plot/df tipo int
teste2 = teste.values
'''removendo valores diferentes de obitos e cura'''
dados  = dados.drop(dados[dados['EVOLUCAO'] > 2 ].index)
dados  = dados.drop(dados[dados['EVOLUCAO'] < 1  ].index)


#numpy array
teste3 = dados.to_numpy()
arr = dados['EVOLUCAO'].values
    
'''tranformando valores em rotulos'''
#tst = teste['EVOLUCAO'].apply(preprocessing.LabelEncoder().fit_transform)        
teste = dados.apply(preprocessing.LabelEncoder().fit_transform)
#primeiro metodo para plot/df tipo int
teste2 = teste.values

# visualização de quantos registros existem por classe
unicos,quantidade = np.unique(teste,return_counts=True)

#instanciando KMeans/ criando agrupamentos
cluster = KMeans(n_clusters=2)
cluster.fit(teste)

#visualização dos centroides(agrupamentos ou clusters anteiormente definidos)
centroides = cluster.cluster_centers_

#visualização dos grupos que cada registro foi associado
previsoes = cluster.labels_

#contagem dos registros por classe
unicos2, quantidade2 = np.unique(previsoes,return_counts = True)

#geração da matriz de contingencia para comparar os grupos com a base de dados
resultados = confusion_matrix(arr,previsoes)
 
'''-----------------------''' 
'''
plt.scatter(teste3[previsoes == 0, 1],teste3[previsoes == 0, 23],
            c = 'green',label = 'Obitos')
plt.scatter(teste3[previsoes == 1, 1],teste3[previsoes == 1, 13],
            c = 'red',label = 'Recuperados')

plt.scatter(teste3[previsoes == 2, 1],teste3[previsoes == 2, 23],
            c = 'blue',label = '')
plt.scatter(teste3[previsoes == 3, 1],teste3[previsoes == 3, 23],
            c = 'black',label = 'Recuperados')
plt.scatter(teste3[previsoes == 4, 1],teste3[previsoes == 4, 23],
            c = 'yellow',label = 'Recuperados')
plt.scatter(teste3[previsoes == 5, 1],teste3[previsoes == 5, 23],
            c = 'pink',label = 'Recuperados')

plt.legend()
'''

'''-------------------------'''

for i in range(len(centroides)):
    centers = centroides[i]
    labels = dados['EVOLUCAO'][i]
    fig = plt.figure(figsize=(7, 5))
    fig.set_tight_layout(True)
    plt.scatter(teste3[:, 1], teste3[:, 23], c=labels,
                s=50, cmap='rainbow');
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
    plt.savefig('kmeans_demo/{}.png'.format(i))













