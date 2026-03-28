import numpy as np
import pandas as pd

def get_ranks_s(selected_datasets,
                scores,
                all_datasets,
                model_indx=np.arange(35),
                return_ranks=False,
                fold_num=30):

    """
    Build fold-wise benchmark ranks from raw score tables and optionally aggregate
    them over a selected dataset subset.
    
    This function is used in two places in the notebook:
    
    1. to precompute per-dataset rank tables from raw benchmark scores;
    2. to obtain the mean model ranks induced by a chosen subset of datasets.
    
    Parameters
    ----------
    selected_datasets : sequence of str
        Dataset identifiers to aggregate over when ``return_ranks`` is ``False``.
    scores : dict
        Mapping ``model_name -> DataFrame`` with one row per dataset and fold
        columns named ``'0'`` ... ``str(fold_num - 1)``.
    all_datasets : sequence of str
        Full ordered list of datasets that should appear in the output tables.
    model_indx : array-like, default=np.arange(35)
        Indices of models to include from ``scores``.
    return_ranks : bool, default=False
        If ``True``, return the complete dictionary of fold-wise rank tables instead
        of aggregating over ``selected_datasets``.
    fold_num : int, default=30
        Number of cross-validation folds stored in each score table.
    
    Returns
    -------
    dict or ndarray
        If ``return_ranks`` is ``True``, returns a dictionary
        ``dataset_name -> DataFrame`` where each fold column contains ranks rather
        than raw scores.
        Otherwise returns a NumPy array with the mean rank of every model across
        the selected datasets and folds.
    
    """
    
    model_indx = np.sort(model_indx)
    
    _table = {dataset : pd.DataFrame(columns=['model'] + list(scores[list(scores.keys())[0]].columns[1:])) for dataset in all_datasets}
   
    for model in np.array(list(scores.keys()))[model_indx]:
        
        metrix = scores[model]
        
        for i in range(len(metrix)):
            
            dataset = metrix.iloc[[i]]["folds:"].item()
            s = metrix.iloc[[i]]
            s.reset_index(drop=True, inplace=True)
            s["folds:"] = model
            s.rename(columns={"folds:": 'model'}, inplace=True)

            if dataset in _table:
                _table[dataset] = pd.concat([_table[dataset], s], ignore_index=True)
                
    _ranks = _table.copy()
    
    for dataset in all_datasets:
        
        t = _table[dataset].copy(deep=True)
        
        for i in range(fold_num):
            t[str(i)][np.argsort(t[str(i)])] = np.arange(len(t[str(i)])) + 1
            
        _ranks[dataset] = t
    
    if return_ranks:
        return _ranks
    
    sum_ =  _ranks[selected_datasets[0]].copy(deep=True)
        
    sum_[[str(i) for i in range(fold_num)]] *= 0
    
    for dataset in selected_datasets:
        sum_[[str(i) for i in range(fold_num)]] += _ranks[dataset].copy(deep=True)

    mean_rank = (sum_[[str(i) for i in range(fold_num)]] / len(selected_datasets)).mean(axis=1).values
    
    return mean_rank


def get_ranks(selected_datasets,
              ranks,
              fold_num=30):
    
    sum_ = ranks[selected_datasets[0]].copy(deep=True)
    sum_[[str(i) for i in range(fold_num)]] *= 0
    
    for dataset in selected_datasets:
        sum_[[str(i) for i in range(fold_num)]] += ranks[dataset].copy(deep=True)[[str(i) for i in range(fold_num)]]

    mean_rank = (sum_[[str(i) for i in range(fold_num)]] / len(selected_datasets)).mean(axis=1).values
    
    return mean_rank