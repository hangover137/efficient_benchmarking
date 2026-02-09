import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor


def a_d_optimality_ind(data: pd.DataFrame,
                       sample_size: int,
                       optimality: str = 'None',
                       iter: int = 100,
                       alpha: float = 0.0001,
                       random_state: int = 42,
                       return_ind = True,
                       **kwargs) -> tuple[np.ndarray[int], np.float64]:
    
    if optimality not in ['a', 'd']:
        raise Exception(f'Incorrect option "{optimality}" for optimality. "a" - for A-optimality, "d" - for D-optimality')
    
    indxs = np.arange(sample_size)
    criteria = .0
    
    np.random.seed(random_state)
    
    for _ in range(iter):
        
        iter_indxs = np.random.choice(np.arange(data.shape[0]), sample_size, replace=False)
        iter_criteria =.0
        
        if optimality == 'a':
            iter_criteria = 1 / np.trace(np.linalg.inv(data.loc[list(iter_indxs)].T.dot(data.loc[list(iter_indxs)])
                                                           + np.eye(data.shape[1]) + alpha))
        if optimality == 'd':
            iter_criteria = np.linalg.det(data.loc[list(iter_indxs)].T.dot(data.loc[list(iter_indxs)]) + 
                                            + alpha * np.eye(data.shape[1]))
        
        for i in range(sample_size):
            
            for j in range(data.shape[0]):
                
                if j in iter_indxs:
                    continue
                
                cur_indxs = iter_indxs.copy()
                cur_indxs[i] = j
                cur_criteria = .0
                
                if optimality == 'a':
                    cur_criteria = 1 / np.trace(np.linalg.inv(data.loc[list(cur_indxs)].T.dot(data.loc[list(cur_indxs)])
                                                           + np.eye(data.shape[1]) + alpha))
                if optimality == 'd':
                    cur_criteria = np.linalg.det(data.loc[list(cur_indxs)].T.dot(data.loc[list(cur_indxs)]) + 
                                                  + alpha * np.eye(data.shape[1]))
                
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

def calc_catboost_a_opt(data,
                        targets,
                        sample,
                        test_size = 100,
                        random_state = 42,
                        n_trees = 20,
                        **kwargs):

    model = CatBoostRegressor(iterations=n_trees, verbose=0, random_state=random_state)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(data.iloc[sample].values)
    y_train = targets[sample]
    
    if np.unique(X_train, axis=0).shape[0] < 2:
        return float('inf')

    if np.all(X_train.std(axis=0) < 1e-11):
        return float('inf')
    
    model.fit(X_train, y_train)

    mins = X_train.min(axis=0)
    maxs = X_train.max(axis=0)

    X_test = np.zeros((test_size, X_train.shape[1]))
    
    for c in range(X_train.shape[1]):
        X_test[:, c] = np.random.uniform(mins[c], maxs[c], size=test_size)
    
    X_test = scaler.fit(X_test)
    tree_vars = []
    
    for i in range(n_trees):
        
        sing_pred = model.predict(X_test, ntree_start=i, ntree_end=i+1)
        
        tree_vars.append(np.var(sing_pred))
        
    crit = np.mean(tree_vars)

    return crit

#------------------------------------------------------------------------------------------------------------

def catboost_a_opt(data,
                   target,
                   sample_size,
                   iter = 10,
                   random_state = 42,
                   return_ind = False,
                   **kwargs):
    
    
    indxs = np.arange(sample_size)
    criteria = float('inf')
    
    np.random.seed(random_state)
    
    for _ in range(iter):
        
        iter_indxs = np.random.choice(np.arange(data.shape[0]), sample_size, replace=False)
        iter_criteria = float('inf')
        
        for i in range(sample_size):
            for j in range(data.shape[0]):
                
                if j in iter_indxs:
                    continue
                
                cur_indxs = iter_indxs.copy()
                cur_indxs[i] = j
                cur_criteria = calc_catboost_a_opt(data, target, cur_indxs)
                
                
                if iter_criteria >= cur_criteria:
                    iter_criteria = cur_criteria
                    iter_indxs[i] = j
        
        if criteria >= iter_criteria:
            criteria = iter_criteria
            indxs = iter_indxs.copy()
    
    if return_ind:
        return indxs
    
    return indxs, criteria