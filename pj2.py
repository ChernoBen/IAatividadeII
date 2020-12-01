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


