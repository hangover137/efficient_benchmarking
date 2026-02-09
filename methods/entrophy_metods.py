import os
import sys
import numpy as np

import torch

from tqdm import trange

from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


if __name__ == '__main__':
    
    from model_training.auto_training import automated_model_training
    
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

else:
    
    from methods.model_training.auto_training import automated_model_training

from utils.get_ranks import get_ranks


def get_top_k_models(ranks, k):

    topk_indices = np.argsort(ranks)[:k]
    
    return topk_indices

#------------------------------------------------------------------------------------------------------------

def binary_entropy(p, eps = 1e-12):

    p = np.clip(p, eps, 1 - eps)
    
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

#------------------------------------------------------------------------------------------------------------

def total_entropy(top_k_counts, s_size):

    if s_size == 0:
        return 0.0
    
    p = top_k_counts / s_size
    
    return np.sum(binary_entropy(p))

#------------------------------------------------------------------------------------------------------------

def max_introphy_ind(chosen_datasets,
                     sample_size: int,
                       iter: int = 100,
                       num_models=35,
                       top_k=3,
                       return_ind=True,
                       **kwargs) -> tuple[np.ndarray[int], np.float64]:
    
    indxs = np.arange(sample_size)
    criteria = .0
    
    for _ in range(iter):
        
        iter_indxs = np.random.choice(np.arange(chosen_datasets.shape[0]), sample_size, replace=False)
        iter_criteria = .0
        top_k_counts = np.zeros(num_models)

        for d in iter_indxs:
            
            ranks = get_ranks([chosen_datasets[d]])
            
            top_k = get_top_k_models(ranks, top_k)
            
            top_k_counts[top_k] += 1
            
        
        iter_criteria = total_entropy(top_k_counts, len(iter_indxs) + 1)
        
        
        for i in range(sample_size):
            for j in range(chosen_datasets.shape[0]):
                
                if j in iter_indxs:
                    continue
                
                top_k_counts = np.zeros(num_models)
                
                cur_indxs = iter_indxs.copy()
                cur_indxs[i] = j
                cur_criteria = .0
                
                for cd in cur_indxs:
            
                    ranks = get_ranks([chosen_datasets[cd]])
                    
                    top_k = get_top_k_models(ranks, top_k)
                    
                    top_k_counts[top_k] += 1
                    
                
                cur_criteria = total_entropy(top_k_counts, len(cur_indxs) + 1)
                
                if iter_criteria <= cur_criteria:
                    iter_criteria = cur_criteria
                    
                    iter_indxs[i] = j
        
        if criteria <= iter_criteria:
            criteria = iter_criteria
            indxs = iter_indxs.copy()
    
    if return_ind:
        return np.array(indxs)
    
    return indxs, criteria

#------------------------------------------------------------------------------------------------------------

def get_ML_entrophy_ind(data,
                        sample_size,
                        epsilon=1e-3,
                        random_state=42,
                        **kwargs):
    
    # train_y = data[:, -35:]
    # train_x = data[:, :-35]
    
    # models_list = []

    # datas = []
    # scalers = []

    # data_1 = train_x
    # data_1 = scaler_x.fit_transform(data_1)
    
    # for i in trange(range(train_y.shape[1]), disable=True):
        
    #     scaler_x = StandardScaler()
    #     data_1 = StandardScaler().fit_transform(data_1)
    #     y = train_y[:, i]
        
    #     data_1 = PCA(n_components=30).fit_transform(data_1)
    #     data_1 = scaler_x.fit_transform(data_1)
        
    #     model, likelihood, _ = automated_model_training(data_1,
    #                                                     y,
    #                                                     max_iter=500,
    #                                                     lr=0.1,
    #                                                     tol=0.01, 
    #                                                     patience=10,
    #                                                     random_state=random_state)
        
    #     models_list.append([model, likelihood])
    #     scalers.append(scaler_x)
    #     datas.append(data_1)

    

    entrophis = []
    
    for dataset_ind in range(data[0][2].shape[0]):
        means = []
        variances = []
    
        for i, (model, likelihood, train_data) in enumerate(data):

            # X_selected = [datas[i][dataset_ind]]
            # X_selected = scalers[i].transform(X_selected)
            
            X_torch = torch.tensor([train_data[dataset_ind]], dtype=torch.float)
            
            model.eval()
            likelihood.eval()
            with torch.no_grad():
                pred = likelihood(model(X_torch))
                mu = pred.mean.item()
                var = pred.variance.item()
            means.append(mu)
            variances.append(var)

        means = np.array(means)
        variances = np.array(variances)
        sigmas = np.sqrt(variances)

        M = means.max() + epsilon

        p_vector = []
        for i in range(len(means)):
            mu_i = means[i]
            sigma_i = sigmas[i]

            if sigma_i > 0:
                p_i_gt_M = 1.0 - norm.cdf((M - mu_i) / sigma_i)
            else:
                p_i_gt_M = float(mu_i > M)

            others_prob = []
            for j in range(len(means)):
                if j != i:
                    mu_j = means[j]
                    sigma_j = sigmas[j]
                    if sigma_j > 0:
                        p_j_lt_M = norm.cdf((M - mu_j) / sigma_j)
                    else:
                        p_j_lt_M = float(mu_j < M)
                    others_prob.append(p_j_lt_M)

            p_i = p_i_gt_M * np.prod(others_prob)
            p_vector.append(p_i)

        p_vector = np.array(p_vector)

        sum_p = p_vector.sum()
        
        if sum_p > 0:
            p_vector /= sum_p
        else:
            p_vector = np.ones_like(p_vector) / len(p_vector)

        entropy = 0.0
        for p_i in p_vector:
            if p_i > 1e-15:
                entropy -= p_i * np.log(p_i)

        result_dict = {
            "probabilities": p_vector,
            "entropy": entropy,
            "means": means,
            "variances": variances,
            "M": M
        }

        entrophis.append(entropy)
    
    
    transp = kwargs.get('transp', False)
    if transp:
        final_res = []
        for size in sample_size:
            final_res.append(np.array(np.argsort(entrophis)[-size:]))
        return final_res
    
    # return np.array(np.argsort(entrophis)[-sample_size:])
    return np.array(np.argsort(entrophis)[:sample_size])
    

#------------------------------------------------------------------------------------------------------------


def get_ML_entrophy_vectors_ind(data,
                        sample_size,
                        epsilon=1e-3,
                        n_samples=1000,
                        random_state=42,
                        **kwargs):

    entrophis = []
    
    for dataset_ind in range(data[0][2].shape[0]):
        means = []
        variances = []
    
        for i, (model, likelihood, train_data) in enumerate(data):

            # X_selected = [datas[i][dataset_ind]]
            # X_selected = scalers[i].transform(X_selected)
            
            X_torch = torch.tensor([train_data[dataset_ind]], dtype=torch.float)
            
            model.eval()
            likelihood.eval()
            with torch.no_grad():
                pred = likelihood(model(X_torch))
                mu = pred.mean.item()
                var = pred.variance.item()
            means.append(mu)
            variances.append(var)

        means = np.array(means)
        variances = np.array(variances)
        sigmas = np.sqrt(variances)
        
        num_models = len(means)
        
        draws = np.zeros((n_samples, num_models), dtype=float)
        
        for i in range(num_models):
            
            if sigmas[i] < 1e-15:
                draws[:, i] = means[i]
            else:
                draws[:, i] = np.random.normal(loc=means[i], scale=sigmas[i], size=n_samples)

        wins = np.zeros(num_models, dtype=int)
        
        max_indices = np.argmax(draws, axis=1)
        
        for idx in max_indices:
            wins[idx] += 1

        p_vector = wins / n_samples

        entropy = 0.0
        for p_i in p_vector:
            if p_i > 1e-15:
                entropy -= p_i * np.log(p_i)

        result_dict = {
            "probabilities": p_vector,
            "entropy": entropy,
            "means": means,
            "variances": variances,
            # "M": M
        }

        entrophis.append(entropy)
    
    transp = kwargs.get('transp', False)
    if transp:
        final_res = []
        for size in sample_size:
            final_res.append(np.array(np.argsort(entrophis)[-size:]))
        return final_res
    
    return np.array(np.argsort(entrophis)[-sample_size:])
    # return np.array(np.argsort(entrophis)[:sample_size])


#------------------------------------------------------------------------------------------------------------


def get_ML_entrophy_vars_ind(data,
                        sample_size,
                        epsilon=1e-3,
                        n_samples=1000,
                        random_state=42,
                        **kwargs):

    entrophis = []
    
    for dataset_ind in range(data[0][2].shape[0]):
        means = []
        variances = []
    
        for i, (model, likelihood, train_data) in enumerate(data):

            # X_selected = [datas[i][dataset_ind]]
            # X_selected = scalers[i].transform(X_selected)
            
            X_torch = torch.tensor([train_data[dataset_ind]], dtype=torch.float)
            
            model.eval()
            likelihood.eval()
            with torch.no_grad():
                pred = likelihood(model(X_torch))
                mu = pred.mean.item()
                var = pred.variance.item()
            means.append(mu)
            variances.append(var)

        means = np.array(means)
        variances = np.array(variances)
        sigmas = np.sqrt(variances)
        
        M = means.max() + epsilon
        
        L = means - 0.5 * sigmas
        R = means + 0.5 * sigmas
        lengths = R - L
        
        def p_gt(i, M):
            if M <= L[i]:
                return 1.0
            elif M >= R[i]:
                return 0.0
            else:
                return (R[i] - M) / lengths[i]

        def p_lt(j, M):
            if M <= L[j]:
                return 0.0
            elif M >= R[j]:
                return 1.0
            else:
                return (M - L[j]) / lengths[j]

        p_vector = []
        for i in range(len(means)):
            p_i_gt_M = p_gt(i, M)

            others_prob = []
            for j in range(len(means)):
                if j == i:
                    continue
                others_prob.append(p_lt(j, M))

            p_i = p_i_gt_M * np.prod(others_prob)
            p_vector.append(p_i)

        p_vector = np.array(p_vector, dtype=float)

        entropy = 0.0
        for p_i in p_vector:
            if p_i > 1e-15:
                entropy -= p_i * np.log(p_i)

        result_dict = {
            "probabilities": p_vector,
            "entropy": entropy,
            "means": means,
            "variances": variances,
            # "M": M
        }

        entrophis.append(entropy)
    
    transp = kwargs.get('transp', False)
    if transp:
        final_res = []
        for size in sample_size:
            final_res.append(np.array(np.argsort(entrophis)[-size:]))
            # final_res.append(np.array(np.argsort(entrophis)[:size]))
        return final_res
    
    return np.array(np.argsort(entrophis)[-sample_size:])


def get_ML_entrophy_mean_variances(data,
                        sample_size,
                        epsilon=1e-3,
                        n_samples=1000,
                        random_state=42,
                        **kwargs):

    entrophis = []
    
    for dataset_ind in range(data[0][2].shape[0]):
        means = []
        variances = []
    
        for i, (model, likelihood, train_data) in enumerate(data):

            # X_selected = [datas[i][dataset_ind]]
            # X_selected = scalers[i].transform(X_selected)
            
            X_torch = torch.tensor([train_data[dataset_ind]], dtype=torch.float)
            
            model.eval()
            likelihood.eval()
            with torch.no_grad():
                pred = likelihood(model(X_torch))
                mu = pred.mean.item()
                var = pred.variance.item()
            means.append(mu)
            variances.append(var)

        means = np.array(means)
        variances = np.array(variances)
        # sigmas = np.sqrt(variances)

        entrophis.append(np.mean(variances))
    
    transp = kwargs.get('transp', False)
    if transp:
        final_res = []
        for size in sample_size:
            final_res.append(np.array(np.argsort(entrophis)[-size:]))
            # final_res.append(np.array(np.argsort(entrophis)[:size]))
        return final_res
    
    return np.array(np.argsort(entrophis)[-sample_size:])