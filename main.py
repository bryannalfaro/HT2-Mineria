#Universidad del Valle de Guatemala
#Mineria de Datos
#HT2 Clustering
#Integrantes
#Bryann Alfaro
#Diego de Jesus
#Julio Herrera

#Referencia: https://www.youtube.com/watch?v=s6PSSzeUMFk&t=1044s

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
from sklearn.decomposition import PCA
import pyclustertend
import random
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
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

'''#Analisis de tendencia a agrupamiento

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
plt.show()'''

'''clusters=  KMeans(n_clusters=3, max_iter=300) #Creacion del modelo
clusters.fit(movies_df_clean) #Aplicacion del modelo de cluster

movies_df_clean['cluster'] = clusters.labels_ #Asignacion de los clusters
print(movies_df_clean.head())

pca = PCA(2)
pca_movies = pca.fit_transform(movies_df_clean)
pca_movies_df = pd.DataFrame(data = pca_movies, columns = ['PC1', 'PC2'])
pca_clust_movies = pd.concat([pca_movies_df, movies_df_clean[['cluster']]], axis = 1)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('PC1', fontsize = 15)
ax.set_ylabel('PC2', fontsize = 15)
ax.set_title('Clusters de peliculas', fontsize = 20)

color_theme = np.array(['red', 'green', 'blue', 'yellow','black'])
ax.scatter(x = pca_clust_movies.PC1, y = pca_clust_movies.PC2, s = 50, c = color_theme[pca_clust_movies.cluster])

plt.show()
print(pca_clust_movies)'''


#Jerarquico
movies_jerarquico = linkage(movies_df_clean,'ward')
#dendograma = sch.dendrogram(movies_jerarquico)
#plt.show()

clusters = fcluster(movies_jerarquico, 10, criterion='distance')
movies_df_clean['cluster jerarquico'] = clusters
print(clusters)
print(movies_df_clean)