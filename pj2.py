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
clss = dados[['EVOLUCAO','NU_IDADE_N']].fillna(0)
'''removendo tudo que nao seja cura e obito'''
clss = clss.drop(clss[clss['EVOLUCAO'] > 2 ].index)
'''removendo valores nao informados'''
clss = clss.drop(clss[clss['EVOLUCAO'] < 1 ].index)

'''organizando kmeans'''
cluster = KMeans(n_clusters = 2)
cluster.fit(clss)
y_kmeans = cluster.predict(clss)

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
plt.scatter(clss[previsoes == 0, 0],clss[previsoes == 0, 1],
            c = 'green',label = 'obtios')
plt.scatter(clss[previsoes == 1, 0],clss[previsoes == 1, 1],
            c = 'red',label = 'recuperados')

plt.legend()

   

