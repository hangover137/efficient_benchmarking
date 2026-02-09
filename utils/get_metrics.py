import numpy as np
import pandas as pd

from scipy.stats import kendalltau, spearmanr
from scipy.stats import chi2_contingency
from scipy.stats import pointbiserialr
from scipy.stats import pearsonr

import xicorpy
import distance_correlation

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mutual_info_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    ndcg_score,
)

def ndcg_from_mean_ranks_1(true_ranks, pred_ranks, k=5, higher_is_better_ranks=True):

    true_ranks = np.asarray(true_ranks, dtype=float)
    pred_ranks = np.asarray(pred_ranks, dtype=float)

    n = len(true_ranks)
    if n == 0:
        return float("nan")
    k = int(min(k, n))

    if higher_is_better_ranks:
        relevance = true_ranks
        scores = pred_ranks
    else:
        m = np.nanmax(true_ranks)
        relevance = (m + 1.0) - true_ranks
        scores = (m + 1.0) - pred_ranks

    relevance = np.nan_to_num(relevance, nan=np.nanmin(relevance))
    scores = np.nan_to_num(scores, nan=np.nanmin(scores))

    return float(ndcg_score(relevance.reshape(1, -1), scores.reshape(1, -1), k=k))



def get_metrics(true_ranks,
                pred_ranks,
                return_simple_metrics=True):

    x = true_ranks
    y = pred_ranks
    
    if return_simple_metrics:
        K=kendalltau(x, y).correlation
        
        return {
                'MAE': mean_absolute_error(true_ranks, pred_ranks),
                'Spearman': spearmanr(true_ranks, pred_ranks).statistic,
                'NDCG@3': ndcg_from_mean_ranks_1(true_ranks, pred_ranks, k=3),
                'NDCG@5': ndcg_from_mean_ranks_1(true_ranks, pred_ranks, k=5),
                # 'NDCG@7': ndcg_from_mean_ranks_1(true_ranks, pred_ranks, k=7),
                # 'NDCG@10': ndcg_from_mean_ranks_1(true_ranks, pred_ranks, k=10),
                # 'NDCG@15': ndcg_from_mean_ranks_1(true_ranks, pred_ranks, k=15),
                'Kendall':np.mean(K)
                }
    
    P=pearsonr(x,y)[0] #Pearson
    S=spearmanr(x, y).correlation #Spearman
    K=kendalltau(x, y).correlation #Kendall
    Mu=mutual_info_score(x, y) #Mutual information
    A=adjusted_mutual_info_score(x, y) #Adjusted mutual information
    N=normalized_mutual_info_score(x, y) #Normalized mutual information
    C=np.sqrt(chi2_contingency(pd.crosstab(x, y))[0] / (pd.crosstab(x, y).sum().sum() * (min(pd.crosstab(x, y).shape) - 1))) #Cramer's V
    B=pointbiserialr(x, y).correlation #Point-Biserial Correlation
    Phi=np.sqrt(chi2_contingency(pd.crosstab(x, y))[0] / pd.crosstab(x, y).sum().sum()) #Phi Coefficient
    Bl=np.mean((x > np.median(x)) == (y > np.median(y))) - np.mean((x > np.median(x)) != (y > np.median(y))) #Bloomquist Beta
    # Cc[i,j]=np.correlate(x, y, mode='full') #Cross-correlation
    R=np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1] #Rank Correlation Ratio
    Me=np.mean((x > np.median(x)) & (y > np.median(y))) + np.mean(
    (x <= np.median(x)) & (y <= np.median(y))) - np.mean((x > np.median(x)) & (y <= np.median(y))) - np.mean((x <= np.median(x)) & (y > np.median(y)))
    
    Xi = xicorpy.compute_xi_correlation(x, y)
    DC = distance_correlation.distance_correlation(x, y)
    
    return {
        'MAE': mean_absolute_error(true_ranks, pred_ranks),
        'MSE': mean_squared_error(true_ranks, pred_ranks),
        # 'Kendall': kendalltau(true_ranks, pred_ranks).statistic,
        'Spearman': spearmanr(true_ranks, pred_ranks).statistic,
        'NDCG@3': ndcg_from_mean_ranks_1(true_ranks, pred_ranks, k=3),
        'NDCG@5': ndcg_from_mean_ranks_1(true_ranks, pred_ranks, k=5),
        'NDCG@7': ndcg_from_mean_ranks_1(true_ranks, pred_ranks, k=7),
        'NDCG@10': ndcg_from_mean_ranks_1(true_ranks, pred_ranks, k=10),
        'NDCG@15': ndcg_from_mean_ranks_1(true_ranks, pred_ranks, k=15),
        'Pearson':np.mean(P),
        # 'Spearman':np.mean(S),
        'Kendall':np.mean(K),
        'Mutual information':np.mean(Mu),
        'Adjusted mutual information':np.mean(A),
        'Normalized mutual information':np.mean(N),
        'Cramers V':np.mean(C),
        'Point-Biserial Correlation':np.mean(B),
        'Phi Coefficient':np.mean(Phi),
        'Bloomquist Beta':np.mean(Bl),
        # 'Cross-correlation':np.mean(Cc),
        'Rank Correlation Ratio':np.mean(R),
        'Xi Correlation': Xi[0][0],
        'Distance Correlation': DC,
        'Me':np.mean(Me)
    }