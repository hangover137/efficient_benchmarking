import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances_argmin_min


def get_more_different_datasets_kmeans(data,
                                       sample_size,
                                       random_state=42,
                                       criteria='max',
                                       **kwargs):

    n_distances = 1 - cosine_similarity(data)

    kmeans = KMeans(
        n_clusters=sample_size,
        random_state=random_state,
        init="k-means++",
        n_init=100,
    )
    
    labels = kmeans.fit_predict(n_distances)
    indxs = []
    
    for label in set(labels):
        
        cluster_indices = np.where(labels == label)[0]
        centr = kmeans.cluster_centers_[label]
        distanses_from_centr = np.linalg.norm(n_distances[cluster_indices] - centr, axis=1)
        label_indx = cluster_indices[np.argmax(distanses_from_centr)]

        if criteria == 'min':
            label_indx = cluster_indices[np.argmin(distanses_from_centr)]
            
        indxs.append(label_indx)

    return np.array(indxs)

#------------------------------------------------------------------------------------------------------------

def k_means_ind(data: pd.DataFrame,
                sample_size: int,
                iter : int = 10,
                random_state:int = 42,
                return_more = False,
                pca_099=False,
                scale_data=True,
                **kwargs):
    
    if pca_099:
        data = np.array(StandardScaler().fit_transform(data))
        data = PCA(0.99).fit_transform(data)  
    if scale_data:
        data = np.array(StandardScaler().fit_transform(data))

    kmeans = KMeans(
        n_clusters=sample_size,
        init="k-means++",
        n_init=10,
        random_state=random_state
    )
    
    kmeans.fit(data)
    kmeans_labels = kmeans.fit_predict(data)
    
    data_indx = np.arange(data.shape[0])
    
    unique_clusters = np.unique(kmeans_labels)
    clusters = {i: data_indx[kmeans_labels == i] for i in unique_clusters}
    
    indxs, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, data)
    
    
    if return_more:
        return indxs.astype(int), clusters, kmeans.cluster_centers_
    
    return np.array(indxs.astype(int))