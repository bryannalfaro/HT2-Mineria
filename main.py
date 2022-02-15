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
from scipy import stats
import seaborn as sns
from collections import Counter
from sklearn import preprocessing
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
import pyclustertend
import random
from clean import *
# movies = pd.read_csv('movies.csv', encoding='unicode_escape')
movies = pd.read_csv('movies.csv', encoding='latin1', engine='python')

'''movies_clean = movies[['popularity', 'budget', 'revenue', 'runtime','genresAmount','productionCoAmount','productionCountriesAmount','releaseDate','voteAvg','voteCount','actorsPopularity','actorsAmount','castWomenAmount','castMenAmount']]

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
plt.show()'''



#Datos con limpieza de variables
movies_clean = movies[['popularity', 'budget', 'revenue', 'runtime','genresAmount','voteAvg','voteCount','castWomenAmount','castMenAmount']]

movies_clean['castMenAmount']=clean_numeric_data(movies_clean["castMenAmount"], True, 200, keep_size=True)["Item"]
movies_clean['castWomenAmount']=clean_numeric_data(movies_clean["castWomenAmount"], True, 200, keep_size=True)["Item"]

print(movies_clean.head().dropna())
print(movies_clean.info())
print(movies_clean.describe())

movies_clean.fillna(0)

#normalizar
movies_clean_norm  = (movies_clean-movies_clean.min())/(movies_clean.max()-movies_clean.min())
print(movies_clean_norm.fillna(0))
movies_df_clean = movies_clean_norm.fillna(0)

print(movies_df_clean.describe())

#Analisis de tendencia a agrupamiento

#Metodo Hopkings

random.seed(200)
print(pyclustertend.hopkins(movies_df_clean, len(movies_df_clean)))

#Grafico VAT e iVAT
x = movies_df_clean.sample(frac=0.1)
pyclustertend.vat(x)
plt.show()
pyclustertend.ivat(x)
plt.show()

# Numero adecuado de grupos
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, max_iter=300)
    kmeans.fit(movies_df_clean)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Grafico de codo')
plt.xlabel('No. Clusters')
plt.ylabel('Puntaje')
plt.show()
