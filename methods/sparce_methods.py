import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


def rand_ind_method(data, size, _model_list=[], **kwargs):
    base_seed = kwargs.get("random_state", None)
    
    if base_seed is None:
        return np.sort(np.random.choice(np.arange(data.shape[0]), size, replace=False))
    else:
        seed = (int(base_seed) * 1000003 + int(size)) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)

        return np.sort(rng.choice(np.arange(data.shape[0]), size, replace=False))
#------------------------------------------------------------------------------------------------------------

def get_more_different_datasets(data,
                                sample_size,
                                # init='point_center',
                                # init_criteria='max',
                                # distance_method='min_max',
                                **kwargs):

    # data = np.array(StandardScaler().fit_transform(data))
    
    defaults = {'init': 'point_center',
            'init_criteria': 'max',
            'distance_method': 'min_max',
            'bandwidth': 0.5,
            'pairwise_dist_type': 'cosine',
            'model_list': [],
            'scale_data': True,
            'pca_099': False
            }
    
    defaults.update(kwargs)
    
    init, init_criteria, distance_method, bandwidth, pairwise_dist_type, model_list, scale_data, pca_099 = defaults.values()
    
    if pca_099:
        data = np.array(StandardScaler().fit_transform(data))
        data = PCA(0.99).fit_transform(data)  
    if scale_data:
        data = np.array(StandardScaler().fit_transform(data))
    
    n_distances = 1 - cosine_similarity(data)
    # n_distances = 1 - cosine_similarity(data) * np.abs(cosine_similarity(data))
    # n_distances = 1 - cosine_similarity(data) ** 2
    
    #-----------------------------------------------------------------------------------------------
    if pairwise_dist_type == 'kde':
        n = data.shape[0]
        n_distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist_sq = np.sum((data[i] - data[j])**2)
                sim_ij = np.exp(- dist_sq / (2 * bandwidth**2))
                d_ij = 1 - sim_ij
                n_distances[i, j] = d_ij
                n_distances[j, i] = d_ij
    #-----------------------------------------------------------------------------------------------
    
    indxs = [np.random.choice(range(len(data)))]
    
    data = np.array(data)
    
    crit = np.argmin
    
    if init_criteria == 'max':
        crit = np.argmax
    
    if init == 'point_center':
        indxs = [crit(np.linalg.norm(data - data.mean(axis=0), axis=1))]
    
    if init == 'cos_center':
        indxs = [crit(cosine_similarity(data, [data.mean(axis=0)]).flatten())]

    for _ in range(sample_size - 1):
        
        lefted_indxs = [i for i in range(len(data)) if i not in indxs]
        
        if distance_method=='min_max':
            min_distances = n_distances[lefted_indxs][:, indxs].min(axis=1)
            next_index = lefted_indxs[np.argmax(min_distances)]
            # print(np.max(min_distances))
        
        if distance_method == 'kde':
            distances_to_chosen = n_distances[lefted_indxs][:, indxs]
            # bandwidth = kwargs.get('kde_bandwidth', 0.5)
            density = np.sum(np.exp(- (distances_to_chosen ** 2) / (2 * bandwidth ** 2)), axis=1)
            next_index = lefted_indxs[np.argmin(-density)]
        
        if distance_method=='max_sqd':
            sqd_mean_distances = np.sqrt((n_distances[lefted_indxs][:, indxs] ** 2).mean(axis=1))
            next_index = lefted_indxs[np.argmax(sqd_mean_distances)]
        
        indxs.append(next_index)

    return np.array(indxs)

#------------------------------------------------------------------------------------------------------------

def get_more_different_datasets_euclid(data, 
                                       sample_size,  
                                       init='point_center',
                                       init_criteria='max',
                                       distance_method='min_max',
                                       pca_099=False,
                                       **kwargs):
    
    if pca_099:
        data = np.array(StandardScaler().fit_transform(data))
        data = PCA(0.99).fit_transform(data)  
    
    data = np.array(StandardScaler().fit_transform(data))
    
    distances = euclidean_distances(data, data)
    
    indxs = [np.random.choice(range(len(data)))]
    
    crit = np.argmin
    if init_criteria == 'max':
        crit = np.argmax

    if init == 'point_center':
        center = data.mean(axis=0)
        dist_to_center = np.linalg.norm(data - center, axis=1)
        indxs = [crit(dist_to_center)]
    
    if init == 'random':
        pass

    for _ in range(sample_size - 1):
        lefted_indxs = [i for i in range(len(data)) if i not in indxs]
        
        if distance_method == 'min_max':
            min_distances = distances[lefted_indxs][:, indxs].min(axis=1)
            next_index = lefted_indxs[np.argmax(min_distances)]
        
        if distance_method == 'max_sqd':
            sqd_distances = (distances[lefted_indxs][:, indxs] ** 2).mean(axis=1)
            next_index = lefted_indxs[np.argmax(np.sqrt(sqd_distances))]

        indxs.append(next_index)

    return np.array(indxs)

#------------------------------------------------------------------------------------------------------------

def get_more_different_datasets_wide(data,
                                sample_size,
                                # init='point_center',
                                # init_criteria='max',
                                # distance_method='min_max',
                                **kwargs):

    defaults = {'init': 'point_center',
                'init_criteria': 'max',
                'distance_method': 'min_max',
                'min_mode': 'classic',
                'alpha': 10,
                'reg_lambda': 0.1,
                'reg_space': 'cosine',
                'model_list': []
                }
    
    defaults.update(kwargs)
    
    init, init_criteria, distance_method, min_mode, alpha, reg_lambda, reg_space, model_list  = defaults.values()
    
    n_distances = 1 - cosine_similarity(data)
    indxs = [np.random.choice(range(len(data)))]
    
    data = np.array(data)
    
    crit = np.argmin
    
    if init_criteria == 'max':
        crit = np.argmax
    
    if init == 'point_center':
        indxs = [crit(np.linalg.norm(data - data.mean(axis=0), axis=1))]
    
    if init == 'cos_center':
        indxs = [crit(cosine_similarity(data, [data.mean(axis=0)]).flatten())]
    
    #----------------------------------------------------------------------------------
    if min_mode == 'regularized':
        
        crit = np.argmin
        
        indxs = [crit(np.linalg.norm(data - data.mean(axis=0), axis=1))]
        
        if reg_space == 'cosine':
            cos_vals = cosine_similarity(data, [data.mean(axis=0)]).flatten()
            all_dist_to_center = 1 - cos_vals
        else:
            all_dist_to_center = np.linalg.norm(data - data.mean(axis=0), axis=1)
    else:
        all_dist_to_center = None 
    #----------------------------------------------------------------------------------

    for _ in range(sample_size - 1):
        
        lefted_indxs = [i for i in range(len(data)) if i not in indxs]
        
        distances = n_distances[lefted_indxs][:, indxs]
        
        
        if min_mode == 'classic':
            dist_scores = distances.min(axis=1)
            
            if distance_method=='max_sqd':
                dist_scores = np.sqrt((n_distances[lefted_indxs][:, indxs]**2).mean(axis=1))
            
        
        if min_mode == 'smooth':
            exps = np.exp(-alpha * distances)
            numer = np.sum(distances * exps, axis=1)
            denom = np.sum(exps, axis=1)
            dist_scores = numer / denom
        
        
        if min_mode == 'regularized':
            # dist_scores = distances.min(axis=1)
            
            # if distance_method=='max_sqd':
            #     dist_scores = np.sqrt((n_distances[lefted_indxs][:, indxs]**2).mean(axis=1))
            
            # reg_term = reg_lambda * all_dist_to_center[lefted_indxs]
            # dist_scores = dist_scores - reg_term
            reg_distances = distances - reg_lambda * all_dist_to_center[lefted_indxs][:, np.newaxis]
            
            if distance_method == 'max_sqd':
                reg_distances = np.sqrt((n_distances[lefted_indxs][:, indxs]**2).mean(axis=1))  # Если здесь требуются изменения, нужно скорректировать отдельно

            dist_scores = reg_distances.min(axis=1)
        
        next_index = lefted_indxs[np.argmax(dist_scores)]
        indxs.append(next_index)

    return np.array(indxs)

#------------------------------------------------------------------------------------------------------------

def select_by_gp_variance(
        data,
        sample_size,
        **kwargs):

    defaults = {
        'scale_data': True,
        'noise': 1e-6
    }
    defaults.update(kwargs)
    scale_data = defaults['scale_data']
    noise = defaults['noise']

    data = np.asarray(data, dtype=float)
    
    if scale_data:
        data = StandardScaler().fit_transform(data)

    K = cosine_similarity(data)

    n = K.shape[0]
    indxs = [np.argmax(np.linalg.norm(data - data.mean(axis=0), axis=1))]
    K_inv = None 

    for _ in range(sample_size - 1):
        
        if not indxs:
            sigma2 = np.ones(n)
            
        K_SS = K[np.ix_(indxs, indxs)] + noise * np.eye(len(indxs))
        K_inv = np.linalg.inv(K_SS)
        
        left = [i for i in range(n) if i not in indxs]
        
        K_xS = K[np.ix_(left, indxs)]
        
        proj = np.sum(K_xS @ K_inv * K_xS, axis=1)
        
        sigma2 = 1.0 - proj
        sigma2 = np.maximum(sigma2, 0.0)

        full_sigma2 = np.zeros(n)
        full_sigma2[left] = sigma2
        full_sigma2[indxs] = -np.inf
        
        sigma2 = full_sigma2

        next_idx = int(np.argmax(sigma2))
        indxs.append(next_idx)

    return np.array(indxs)
