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


dados = pd.read_csv('filtered_dataset', sep= ';') 