#Universidad del Valle de Guatemala
#Mineria de Datos
#HT2 Clustering
#Integrantes
#Bryann Alfaro
#Diego de Jesus
#Julio Herrera

#Referencia: https://www.youtube.com/watch?v=s6PSSzeUMFk&t=1044s
#https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html

from math import ceil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from collections import Counter
from sklearn import preprocessing
from sklearn import datasets
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.decomposition import PCA
import pyclustertend
import random
import sklearn.mixture as mixture
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from clean import *
import matplotlib.cm as cm
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



#Datos con limpieza de variables
movies_clean = movies[['popularity', 'budget', 'revenue', 'runtime','genresAmount','voteAvg','voteCount','castWomenAmount','castMenAmount']]

movies_clean['castMenAmount']=clean_numeric_data(movies_clean["castMenAmount"], True, 200, keep_size=True)["Item"]
movies_clean['castWomenAmount']=clean_numeric_data(movies_clean["castWomenAmount"], True, 200, keep_size=True)["Item"]

#print(movies_clean.head().dropna())
#print(movies_clean.info())
#print(movies_clean.describe())

movies_clean.fillna(0)

#normalizar
movies_clean_norm  = (movies_clean-movies_clean.min())/(movies_clean.max()-movies_clean.min())
#print(movies_clean_norm.fillna(0))
movies_df_clean = movies_clean_norm.fillna(0)

#print(movies_df_clean.describe())

#Analisis de tendencia a agrupamiento

#Metodo Hopkings

random.seed(200)
print(pyclustertend.hopkins(movies_df_clean, len(movies_df_clean)))

#Grafico VAT e iVAT
x = movies_df_clean.sample(frac=0.1) # 0.5 para el 50% de los datos como se mostró en el documento
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


#Kmeans
clusters=  KMeans(n_clusters=3, max_iter=300) #Creacion del modelo
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

# Datos individuales de cada cluster
clusters_df = pd.DataFrame()
clusters_df = movies_clean
clusters_df['cluster'] = clusters.labels_
pd.set_option('display.max_rows', 500)
print(clusters_df[clusters_df.cluster == 0])
print(clusters_df[clusters_df.cluster == 0].describe())
print(clusters_df[clusters_df.cluster == 1])
print(clusters_df[clusters_df.cluster == 1].describe())
print(clusters_df[clusters_df.cluster == 2])
print(clusters_df[clusters_df.cluster == 2].describe())

#Jerarquico
movies_jerarquico = linkage(movies_df_clean,'ward')
#dendograma = sch.dendrogram(movies_jerarquico)
#plt.show()

clusters = fcluster(movies_jerarquico, 10, criterion='distance')
movies_df_clean['cluster jerarquico'] = clusters
print(clusters)
print(movies_df_clean)

#Mixtures of Gaussians
gaussian_movies = mixture.GaussianMixture(n_components=3).fit(movies_df_clean)
labels = gaussian_movies.predict(movies_df_clean)
movies_df_clean['cluster gaussian'] = labels
print(movies_df_clean.head())
pca = PCA(2)
pca_movies = pca.fit_transform(movies_df_clean)
pca_movies_df = pd.DataFrame(data = pca_movies, columns = ['PC1', 'PC2'])
pca_clust_movies = pd.concat([pca_movies_df, movies_df_clean[['cluster gaussian']]], axis = 1)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('PC1', fontsize = 15)
ax.set_ylabel('PC2', fontsize = 15)
ax.set_title('Clusters de peliculas', fontsize = 20)

color_theme = np.array(['red', 'green', 'blue', 'yellow','black'])
ax.scatter(x = pca_clust_movies.PC1, y = pca_clust_movies.PC2, s = 50, c = labels)

plt.show()

#KMEANS Siuellete
clusterer = KMeans(n_clusters=3, random_state=10)
cluster_labels = clusterer.fit_predict(movies_df_clean)
movies_df_clean['cluster Kmeans'] = cluster_labels

pca = PCA(2)
pca_movies = pca.fit_transform(movies_df_clean)
pca_movies_df = pd.DataFrame(data = pca_movies, columns = ['PC1', 'PC2'])
pca_clust_movies = pd.concat([pca_movies_df, movies_df_clean[['cluster Kmeans']]], axis = 1)

#Mixtures of Gaussians Siuellete
gaussian_movies = mixture.GaussianMixture(n_components=3).fit(movies_df_clean)
cluster_labels = gaussian_movies.predict(movies_df_clean)
movies_df_clean['cluster gaussian'] = cluster_labels
pca = PCA(2)
pca_movies = pca.fit_transform(movies_df_clean)
pca_movies_df = pd.DataFrame(data = pca_movies, columns = ['PC1', 'PC2'])
pca_clust_movies = pd.concat([pca_movies_df, movies_df_clean[['cluster gaussian']]], axis = 1)

#Jerarquico Siuellete
hc = AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
cluster_labels = hc.fit_predict(movies_df_clean)
movies_df_clean['cluster jerarquico'] = cluster_labels


pca = PCA(2)
pca_movies = pca.fit_transform(movies_df_clean)
pca_movies_df = pd.DataFrame(data = pca_movies, columns = ['PC1', 'PC2'])
pca_clust_movies = pd.concat([pca_movies_df, movies_df_clean[['cluster jerarquico']]], axis = 1)

#Silueta

# The silhouette_score gives the average value for all the samples.
# This gives a perspective into the density and separation of the formed
# clusters
silhouette_avg = silhouette_score(movies_df_clean, cluster_labels)
print("For clusters =", 3, "The average silhouette_score is :", silhouette_avg)

# Compute the silhouette scores for each sample
sample_silhouette_values = silhouette_samples(movies_df_clean, cluster_labels)



fig, (ax) = plt.subplots(1)
fig.set_size_inches(18, 7)

# The 1st subplot is the silhouette plot
# The silhouette coefficient can range from -1, 1 but in this example all
# lie within [-0.1, 1]
ax.set_xlim([-0.1, 1])
# The (n_clusters+1)*10 is for inserting blank space between silhouette
# plots of individual clusters, to demarcate them clearly.
ax.set_ylim([0, len(pca_clust_movies) + (3 + 1) * 10])
ax.scatter(x = pca_clust_movies.PC1, y = pca_clust_movies.PC2, marker="$%d$" % 3, alpha=1, s=50, edgecolor="k")

y_lower = 10
for i in range(3):
    # Aggregate the silhouette scores for samples belonging to
    # cluster i, and sort them
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.nipy_spectral(float(i) / 3)
    ax.fill_betweenx(
        np.arange(y_lower, y_upper),
        0,
        ith_cluster_silhouette_values,
        facecolor=color,
        edgecolor=color,
        alpha=0.7,
    )

    # Label the silhouette plots with their cluster numbers at the middle
    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

    # Compute the new y_lower for next plot
    y_lower = y_upper + 10  # 10 for the 0 samples

ax.set_title("The silhouette plot for 3 clusters.")
ax.set_xlabel("The silhouette coefficient values")
ax.set_ylabel("Cluster label")

# The vertical line for average silhouette score of all the values
ax.axvline(x=silhouette_avg, color="red", linestyle="--")

ax.set_yticks([])  # Clear the yaxis labels / ticks
ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

plt.show()