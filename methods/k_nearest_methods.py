import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


def select_k_closest_datasets_by_ranks(ranks_df,
                                       sample_size,
                                       **kwargs):

    centr = ranks_df.mean(axis=0).values

    distances = np.linalg.norm(ranks_df.values - centr, axis=1)

    closest_ind = np.argsort(distances)[:sample_size]

    return np.array(closest_ind)

#------------------------------------------------------------------------------------------------------------

def select_k_closest_datasets_cos(ranks_df,
                                  sample_size,
                                  **kwargs):

    c = ranks_df.mean(axis=0).values

    c = c.reshape(1, -1)

    cos_sim = cosine_similarity(ranks_df.values, c).flatten()

    dist = 1 - cos_sim

    closest_ind = np.argsort(dist)[:sample_size]

    return np.array(closest_ind)

#------------------------------------------------------------------------------------------------------------

def select_k_closest_in_cosine_space(ranks_df,
                                  sample_size,
                                  **kwargs):

    data_array = ranks_df.values 

    cos_sim_matrix = cosine_similarity(data_array, data_array)
    cos_dist_matrix = 1 - cos_sim_matrix

    dist_sums = cos_dist_matrix.sum(axis=1)
    cent_ind = np.argmin(dist_sums)

    cent = cos_dist_matrix[cent_ind]

    sorted_ind = np.argsort(cent)

    selected_k = sorted_ind[:sample_size]

    return np.array(selected_k)