import numpy as np
import pandas as pd

def get_ranks_s(selected_datasets,
                scores,
                all_datasets,
                model_indx=np.arange(35),
                return_ranks=False,
                fold_num=30):
    
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