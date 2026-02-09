import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import mutual_info_regression

import torch
import gpytorch

from tqdm.notebook import tqdm
from datetime import datetime
import re

import warnings
warnings.simplefilter('ignore')

if __name__ == '__main__':
    
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    

from methods.model_training.auto_training import automated_model_training, ExactGPModel, train_model
from utils.plotting.plot_results import *
from utils.get_ranks import get_ranks, get_ranks_s
from utils.get_metrics import get_metrics



def average_metr(fn,
                 args,
                 data,
                 indexes_list,
                 sample_size,
                 test_ind,
                 datasets,
                 ranks,
                 add_data,
                 models_bench_data,
                 iter=50,
                 models_bench=False,
                 num_models=35):
    
    iter_vals = {}
    
    for i in range(iter):
        
        iter_data = data[indexes_list[i]]
        iter_models_list = np.arange(num_models)
        
        if models_bench:
            iter_data = data
            iter_models_list = indexes_list[i]
        
        if len(add_data[0]) != 0:
            iter_data = add_data[0][i][indexes_list[i]]
        
        if len(add_data[1]) != 0:
            iter_data = add_data[1][i]
        
        if len(add_data[2]) != 0:
            iter_data = add_data[2][i]
        
        indxs = fn(iter_data,
                   sample_size - 1 * (sample_size == 110),
                   model_list=iter_models_list,
                   **args)
        if not models_bench:
            indxs = indexes_list[i][indxs]
        repr_simple = datasets[indxs].squeeze()
        
        if models_bench:
            ranks_simple = get_ranks(repr_simple, models_bench_data[0][i])
            metr = get_metrics(get_ranks(datasets.squeeze(), models_bench_data[0][i]), ranks_simple)
        else:
            ranks_simple = get_ranks(repr_simple, ranks)

        if np.issubdtype(type(test_ind[0]), np.integer) and not models_bench:
            metr = get_metrics(get_ranks(datasets[test_ind].squeeze(), ranks),
                               ranks_simple)
        elif not models_bench:
            metr = get_metrics(get_ranks(test_ind, ranks),
                               ranks_simple)
        
        for key, val in metr.items():
            if key not in iter_vals:
                iter_vals[key] = []
            iter_vals[key].append(val)
            
    aver_metr = {}
    
    for key, vals in iter_vals.items():
        
        vals = np.array(vals)
        
        mean_val = vals.mean()
        
        q025 = np.quantile(vals, 0.025)
        q975 = np.quantile(vals, 0.975)
        
        aver_metr[key] = {
            "mean": mean_val,
            "q2.5": q025,
            "q97.5": q975
        }
    
    return aver_metr

# -------------------------------------------------------------------------------------------------------------------------

def average_metr_transp(fn,
                        args,
                        data,
                        indexes_list,
                        sizes,
                        test_ind,
                        datasets,
                        ranks,
                        add_data,
                        models_bench_data,
                        label,
                        iter=50,
                        models_bench=False,
                        num_models=35):
    
    iter_vals_list = [{} for _ in range(len(sizes))]
    
    for i in tqdm(range(iter), desc=label):
        
        iter_data = data[indexes_list[i]]!
        
        iter_models_list = np.arange(num_models)
        
        if models_bench:
            iter_data = data
            iter_models_list = indexes_list[i]
        
        if len(add_data[0]) != 0:
            iter_data = add_data[0][i][indexes_list[i]]
        
        if len(add_data[1]) != 0:
            iter_data = add_data[1][i]
        
        if len(add_data[2]) != 0:
            iter_data = add_data[2][i]
        
        args.update({'transp': True})
        
        indxs_list = fn(iter_data,
                        sizes,
                        model_list=iter_models_list,
                        **args)
        
        for j, indxs_i in enumerate(indxs_list):
            if not models_bench:
                indxs = indexes_list[i][indxs_i]
            else:
                indxs = indxs_i
            
            repr_simple = datasets[indxs].squeeze()
            
            if not models_bench:
                ranks_simple = get_ranks(repr_simple, ranks)
            else:
                ranks_simple = get_ranks(repr_simple, models_bench_data[0][i])
                metr = get_metrics(get_ranks(datasets.squeeze(), models_bench_data[0][i]), ranks_simple)
            
            if np.issubdtype(type(test_ind[0]), np.integer) and not models_bench:
                metr = get_metrics(get_ranks(datasets[test_ind].squeeze(), ranks), ranks_simple)
            elif not models_bench:
                metr = get_metrics(get_ranks(test_ind, ranks), ranks_simple)
            
            for key, val in metr.items():
                if key not in iter_vals_list[j]:
                    iter_vals_list[j][key] = []
                iter_vals_list[j][key].append(val)
            
    metrics = []
    
    for iter_vals in iter_vals_list:
        aver_metr = {}
        
        for key, vals in iter_vals.items():
            
            vals = np.array(vals)
            
            mean_val = vals.mean()
            
            q025 = np.quantile(vals, 0.025)
            q975 = np.quantile(vals, 0.975)
            
            aver_metr[key] = {
                "mean": mean_val,
                "q2.5": q025,
                "q97.5": q975
            }
        
        metrics.append(aver_metr)
    
    return metrics

# -------------------------------------------------------------------------------------------------------------------------

def eval_method_mean(fn,
                     args,
                     transp,
                     test_iter,
                     data,
                     indexes_list,
                     sizes,
                     datasets,
                     ranks,
                     test_ind,
                     label,
                     add_data,
                     models_bench,
                     num_models,
                     models_bench_data):
    
    metrics = []

    if transp:
        metrics = average_metr_transp(fn,
                                      args,
                                      data,
                                      indexes_list,
                                      sizes,
                                      test_ind,
                                      datasets,
                                      ranks,
                                      add_data,
                                      models_bench_data,
                                      label,
                                      iter=test_iter,
                                      models_bench=models_bench,
                                      num_models=num_models)
        
        metrics[0]['sizes'] = sizes
        
        return metrics
    
    for n_clusters in tqdm(sizes, desc=label):
        
        metr = average_metr(fn,
                            args,
                            data,
                            indexes_list,
                            n_clusters - 1 * (n_clusters == 110),
                            test_ind,
                            datasets,
                            ranks,
                            add_data,
                            models_bench_data,
                            iter=test_iter,
                            models_bench=models_bench,
                            num_models=num_models)
        
        metr['sizes'] = sizes
        
        metrics.append(metr)
    
    return metrics

def get_entrohpy_data(data,
                      load_models,
                      save_models,
                      save_models_path,
                      models_prefix,
                      test_iter,
                      train_inxs_list,
                      label,
                      save_models_hist_path,
                      time_save,
                      random_state,
                      recsys_bench,
                      num_models,
                      models_bench):
    
    entr_data = []
    
    if load_models and os.path.exists(os.path.join(save_models_path, f"{models_prefix}models_list_{test_iter-1}.pkl")):
        print(f'Loading models for {label}...')
        
    dis = load_models and os.path.exists(os.path.join(save_models_path, f"{models_prefix}models_list_{test_iter-1}.pkl"))
    
    for _ind, train_ind in enumerate(tqdm(train_inxs_list, desc=label + ' model training part', disable=dis)):
        
        train_y = data[:, -num_models:][train_ind]
        train_x = data[:, :-num_models][train_ind]
        
        if models_bench:
            train_y = data[:, -num_models:][:, train_ind]
            train_x = data[:, :-num_models]
        
        _data_1 = train_x
        
        if 'PCA' in models_prefix:
            data_pd = pd.DataFrame(train_x)
            _X = StandardScaler().fit_transform(data_pd)
            
            if '30' in models_prefix:
                _data_1 = PCA(n_components=30).fit_transform(_X)
            if '099' in models_prefix:
                _data_1 = PCA(n_components=0.99).fit_transform(_X)
                
            _data_1 = np.array(StandardScaler().fit_transform(_data_1))
        
        models_list = []
        
        if load_models and os.path.exists(os.path.join(save_models_path, f"{models_prefix}models_list_{_ind}.pkl")):
            
            with open(os.path.join(save_models_path, f"{models_prefix}models_list_{_ind}.pkl"), 'rb') as f:
                models_list, train_ind = pickle.load(f)
                
            entr_data.append(models_list)
            continue
            
        
        for i in tqdm(range(train_y.shape[1]), disable=True):
            
            _y = train_y[:, i]
            
            if 'info' in models_prefix:
                mi = mutual_info_regression(train_x, _y, random_state=random_state)
                sorted_indices = np.argsort(mi)[::-1]
                k = 30
                top_k_indices = sorted_indices[:k]
                _data_1 = train_x[:, top_k_indices]
                _data_1 = np.array(StandardScaler().fit_transform(_data_1))
            
            if 'simple' not in models_prefix:
                _model, _likelihood, _ = automated_model_training(_data_1,
                                                                    _y,
                                                                    max_iter=3000,
                                                                    lr=0.1,
                                                                    impr_part=0.01,
                                                                    patience=10,
                                                                    random_state=random_state)
            
            else:
                _likelihood = gpytorch.likelihoods.GaussianLikelihood()
                _torch_data = torch.from_numpy(_data_1).float()
                _torch_y = torch.from_numpy(_y).float()
                _model = ExactGPModel(_torch_data, _torch_y, _likelihood)
                _model, _likelihood = train_model(_torch_data,
                                                    _torch_y,
                                                    _model,
                                                    _likelihood)
                
            models_list.append([_model, _likelihood, _data_1])
            
            
        if save_models:
            with open(os.path.join(save_models_path, f"{models_prefix}models_list_{_ind}.pkl"), 'wb') as f:
                pickle.dump((models_list, train_ind), f)
        
        if not save_models:
            with open(os.path.join(save_models_hist_path, f"{models_prefix}models_list_{_ind}_{time_save}.pkl"), 'wb') as f:
                pickle.dump((models_list, train_ind), f)

        entr_data.append(models_list)
        
    if load_models and os.path.exists(os.path.join(save_models_path, f"{models_prefix}models_list_{test_iter-1}.pkl")):
        print(f'Models for {label} successfully loaded')
    
    return entr_data

def get_lasso_data(data,
                   train_inxs_list,
                   label,
                   random_state):
    
    lasso_feat = []
    for train_ind in tqdm(train_inxs_list, desc=f'{label} lasso part'):
            data_pd = pd.DataFrame(data[:, :-1][train_ind])
            
            _X = StandardScaler().fit_transform(data_pd)
            _y = data[:, -1][train_ind]
            
            lasso = LassoCV(cv=5, random_state=random_state)
            lasso.fit(_X, _y)
            
            coef = pd.Series(lasso.coef_, index=data_pd.columns)
            _selected_features_lasso = coef.index.tolist()
            
            if coef.sum() != 0:
                _selected_features_lasso = coef[coef != 0].index.tolist()
                
            data_1 = pd.DataFrame(data[:, :-1])[_selected_features_lasso].values
            # print(_selected_features_lasso)
            lasso_feat.append(data_1)
    
    return lasso_feat

def get_models_base(datasets,
                    scores,
                    all_datasets,
                    ratio,
                    num_models,
                    load_ranks,
                    test_iter,
                    save_ranks,
                    save_check_path,
                    random_state):
    
        np.random.seed(random_state)
        
        train_indxs_list = []
        for _ in range(test_iter):
            train_indxs_list.append(np.sort(np.random.choice(np.arange(num_models, dtype='int16'),
                                            int(num_models * ratio),
                                            replace=False)))
        
        
        ranks_list = []
        
        if load_ranks and os.path.exists(os.path.join(save_check_path, f"ranks_saved.pkl")):
            with open(os.path.join(save_check_path, f"ranks_saved.pkl"), 'rb') as f:
                ranks_list = pickle.load(f)
        else:
            for train_indx in tqdm(train_indxs_list, desc='Calculating Ranks'):
                ranks_list.append(get_ranks_s(datasets, scores, all_datasets, model_indx=train_indx, return_ranks=True))
        
        if save_ranks:
            with open(os.path.join(save_check_path, f"ranks_saved.pkl"), 'wb') as f:
                pickle.dump(ranks_list, f)
        
        return train_indxs_list, ranks_list

# -------------------------------------------------------------------------------------------------------------------------

def get_ranks_data(train_inxs_list,
                   datasets,
                   ranks_list):
    
    ranks_data = []
        
    for i in range(len(train_inxs_list)):
    
        _ranks_aggr = pd.DataFrame(columns=datasets)
        
        for d in datasets:
            _ranks_aggr[d] = ranks_list[i][d].drop(columns=['model']).mean(axis=1)

        ranks_data.append(_ranks_aggr.transpose().values)
    
    return ranks_data

# -------------------------------------------------------------------------------------------------------------------------

# save_plot_path=os.path.join('results', 'full_testing_d_res_recsys'),
# save_check_path=os.path.join('checkpoints', 'full_testing_ch_recsys'),
# save_history_path=os.path.join('checkpoints', 'history_recsys'),
# save_models_path = os.path.join('checkpoints', 'entrophy_models', 'checkpoints_recsys'),
# save_models_hist_path = os.path.join('checkpoints', 'entrophy_models', 'history_recsys'),
# save_res_path=os.path.join('results', 'datasets_testing_res_recsys'),

# save_plot_path=os.path.join('results', 'full_testing_models_res'),
# save_check_path=os.path.join('checkpoints', 'full_testing_ch_models'),
# save_history_path=os.path.join('checkpoints', 'history_models'),
# save_models_path = os.path.join('checkpoints', 'entrophy_models', 'checkpoints_models'),
# save_models_hist_path = os.path.join('checkpoints', 'entrophy_models', 'history_models'),
# save_res_path=os.path.join('results', 'datasets_testing_res_models'),

# -------------------------------------------------------------------------------------------------------------------------

def testing_pipeline(datasets,
                     metods_data_list,
                    #  labels,
                     ranks,
                     scores,
                     all_datasets,
                     ratio = 0.5,
                     num_models=35,
                     test_iter=50,
                     save_plot_path=os.path.join('results', 'full_testing_d_res'),
                     save_res_path=os.path.join('results', 'datasets_testing_res'),
                     save_check_path=os.path.join('checkpoints', 'full_testing_ch'),
                     save_history_path=os.path.join('checkpoints', 'history'),
                     save_models_path = os.path.join('checkpoints', 'entrophy_models', 'checkpoints'),
                     save_models_hist_path = os.path.join('checkpoints', 'entrophy_models', 'history'),
                     models_prefix='info_',
                     random_state=42,
                     load_res=True,
                     save_models=True,
                     load_models=True,
                     save_checpoints=True,
                     save_results=False,
                     test_datasets=None,
                     save_ranks=True,
                     load_ranks=True,
                     recsys_bench=False,
                     models_bench=False
                     ):
    
    
    np.random.seed(random_state)
    
    train_inxs_list = []
    for _ in range(test_iter):
        train_inxs_list.append(np.sort(np.random.choice(np.arange(datasets.shape[0], dtype='int16'),
                                           int(datasets.shape[0] * ratio),
                                           replace=False)))
    
    models_bench_data = []
    if models_bench:
        _all_ranks = get_ranks_s(datasets, scores, all_datasets,
                                 return_ranks=True, model_indx=np.arange(num_models))
        train_inxs_list, ranks_list = get_models_base(datasets=datasets,
                                                     scores=scores,
                                                     all_datasets=all_datasets,
                                                     ratio=ratio,
                                                     num_models=num_models,
                                                     load_ranks=load_ranks,
                                                     test_iter=test_iter,
                                                     save_ranks=save_ranks,
                                                     save_check_path=save_check_path,
                                                     random_state=random_state)
        
        models_bench_data = [ranks_list, _all_ranks]
    
    test_indxs = np.arange(datasets.shape[0])
    
    if test_datasets is not None:
        test_indxs = test_datasets
    if models_bench:
        test_indxs = np.arange(num_models)
    
    metr_list = []
    labels = []
    
    time_save = re.sub(r'[-:\s.]', '_', str(datetime.now()))
    
    
    for (method, data, sizes, need_l, need_m, args, transp, models_in_data, label) in metods_data_list:
        
        
        if load_res and os.path.exists(os.path.join(save_check_path, f"{label}.pkl")):
            print(f"Loading {label} results...") 
            
            with open(os.path.join(save_check_path, f"{label}.pkl"), 'rb') as f:
                metr = pickle.load(f)
            
            print(f"{label} results successfully loaded")  
             
            labels.append(label) 
            metr_list.append(metr)
            continue
        
        
        ranks_data = []
        if models_in_data and models_bench:
            ranks_data = get_ranks_data(train_inxs_list,
                                        datasets,
                                        ranks_list)
        
        lasso_feat = []
        if need_l:
            get_lasso_data(data=data,
                           train_inxs_list=train_inxs_list,
                           label=label,
                           random_state=random_state)
        
        entr_data = []
        if need_m:
            entr_data = get_entrohpy_data(data=data,
                                         load_models=load_models,
                                         save_models=save_models,
                                         save_models_path=save_models_path,
                                         models_prefix=models_prefix,
                                         test_iter=test_iter,
                                         train_inxs_list=train_inxs_list,
                                         label=label,
                                         save_models_hist_path=save_models_hist_path,
                                         time_save=time_save,
                                         random_state=random_state,
                                         recsys_bench=recsys_bench,
                                         num_models=num_models,
                                         models_bench=models_bench)        


        metr = eval_method_mean(fn=method,
                                args=args,
                                transp=transp,
                                test_iter=test_iter,
                                data=data,
                                indexes_list=train_inxs_list,
                                sizes=sizes,
                                datasets=datasets,
                                ranks=ranks,
                                test_ind=test_indxs,
                                label=label,
                                add_data=[lasso_feat, entr_data, ranks_data],
                                models_bench=models_bench,
                                num_models=num_models,
                                models_bench_data=models_bench_data)
        
        if save_checpoints:
            with open(os.path.join(save_check_path, f"{label}.pkl"), 'wb') as f:
                pickle.dump(metr, f)
        
        if not save_checpoints:
            with open(os.path.join(save_history_path, f"{label}_{time_save}.pkl"), 'wb') as f:
                pickle.dump(metr, f)
        
        
        metr_list.append(metr)
        labels.append(label)
    
    
    plot_results(metr_list=metr_list,
                 labels=labels,
                 save_plot_path=save_plot_path,
                 save_results=save_results,
                 save_res_path=save_res_path,
                 time_save=time_save)
