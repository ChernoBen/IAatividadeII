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
cluster = KMeans(n_clusters=2)
cluster.fit(teste)

#visualização dos centroides(agrupamentos ou clusters anteiormente definidos)
centroides = cluster.cluster_centers_

#visualização dos grupos que cada registro foi associado
previsoes = cluster.labels_

#contagem dos registros por classe
unicos2, quantidade2 = np.unique(previsoes,return_counts = True)

#geração da matriz de contingencia para comparar os grupos com a base de dados
resultados = confusion_matrix(teste['EVOLUCAO'],previsoes)

'''--------------------------''' 
centers = []
labels = [] 
for i in range(len(centroides)):
    centers.append(centroides[i])
    labels.append(teste['EVOLUCAO'][i])
    fig = plt.figure(figsize=(7, 5))
    fig.set_tight_layout(True)
    plt.scatter(teste['EVOLUCAO'][:], teste['NU_IDADE_N'][:], c=labels,
                s=50, cmap='rainbow');
 