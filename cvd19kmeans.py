# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 19:01:57 2020

@author: Benjamim
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
import seaborn as sb



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
arr = []
gabarito = dados['EVOLUCAO']
X = np.array(dados.drop('EVOLUCAO',axis=1))

for item in X[:,14]:
    if item == 'S':
        arr.append(1)
    else:
        arr.append(0)
X[:,14] = arr        
        
kmeans = KMeans(n_clusters=4,random_state=0)
kmeans.fit(X)
'''visualizar labels'''
kmeans.labels_
lb = kmeans.cluster_centers_
'''adicionando resultados a uma nova coluna na base original'''
dados['result'] = kmeans.labels_

