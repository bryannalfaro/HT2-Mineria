#Universidad del Valle de Guatemala
#Mineria de Datos
#HT2 Clustering
#Integrantes
#Bryann Alfaro
#Diego de Jesus
#Julio Herrera

from math import ceil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from scipy import stats
import seaborn as sns
from collections import Counter
# movies = pd.read_csv('movies.csv', encoding='unicode_escape')
movies = pd.read_csv('movies.csv', encoding='latin1', engine='python')