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
'''
1-cura
2-obito
3-obito outras causas
9-ignorado

tranformar cada instancia de cada coluna em labels 
''' 

dados = pd.read_csv('https://raw.githubusercontent.com/ChernoBen/IAatividadeII/main/filtered_dataset', sep= ';')

''' .fillna(0) troca valores Nan por 0'''
clss = dados[['NU_IDADE_N','EVOLUCAO']].fillna(0)
'''removendo tudo que nao seja cura e obito'''
clss = clss.drop(clss[clss['EVOLUCAO'] > 2 ].index)
'''removendo valores nao informados'''
clss = clss.drop(clss[clss['EVOLUCAO'] < 1 ].index)

# visualização de quantos registros existem por classe
unicos,quantidade = np.unique(clss,return_counts=True)

cluster = KMeans(n_cluster)