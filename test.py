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

tranformar cada instancia de cada coluna em labels 
''' 

dados = pd.read_csv('new_dataset', sep= ';')

'''tranformando valores em rotulos'''        
teste = dados.apply(preprocessing.LabelEncoder().fit_transform)

# visualização de quantos registros existem por classe
unicos,quantidade = np.unique(teste,return_counts=True)

