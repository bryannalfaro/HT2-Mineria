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
from clean import *
# movies = pd.read_csv('movies.csv', encoding='unicode_escape')
movies = pd.read_csv('movies.csv', encoding='latin1', engine='python')

movies_clean = movies[['popularity', 'budget', 'revenue', 'runtime','genresAmount','productionCoAmount','productionCountriesAmount','releaseDate','voteAvg','voteCount','actorsPopularity','actorsAmount','castWomenAmount','castMenAmount']]

#preprocesamiento
movies_clean['castMenAmount']=clean_numeric_data(movies_clean["castMenAmount"], True, 313, keep_size=True)["Item"]
movies_clean['castWomenAmount']=clean_numeric_data(movies_clean["castWomenAmount"], True, 313, keep_size=True)["Item"]
corr_data = movies_clean.iloc[:,:]
mat_correlation=corr_data.corr() # se calcula la matriz , usando el coeficiente de correlacion de Pearson
plt.figure(figsize=(16,10))
#Realizando una mejor visualizacion de la matriz
sns.heatmap(mat_correlation,annot=True,cmap='BrBG')
plt.title('Matriz de correlaciones  para la base Movies')
plt.tight_layout()
plt.show()


#Analisis de tendencia a agrupamiento

#Metodo Hopkings