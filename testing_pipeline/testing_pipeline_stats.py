import os
import sys
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import mutual_info_regression

from scipy.stats import friedmanchisquare, wilcoxon, rankdata

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


def _save_summary_ci_csv(metrics: list, out_csv: str):

    sizes = metrics[0]["sizes"]
    rows = []
    for j, k in enumerate(sizes):
        m_j = metrics[j]
        for metric_name, v in m_j.items():
            if metric_name in ("sizes", "label", "repr"):
                continue
            rows.append({
                "size": int(k),
                "metric": metric_name,
                "mean": float(v["mean"]),
                "q2.5": float(v["q2.5"]),
                "q97.5": float(v["q97.5"]),
            })
            
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def _save_raw_csv(raw_by_size: list, sizes, out_csv: str):
    rows = []
    for j, k in enumerate(sizes):
        raw_j = raw_by_size[j]
        for metric_name, vals in raw_j.items():
            for it, val in enumerate(vals):
                rows.append({
                    "size": int(k),
                    "metric": metric_name,
                    "iter": int(it),
                    "value": float(val),
                })
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def _load_raw_csv_to_structure(csv_path: str, sizes):
    df = pd.read_csv(csv_path)
    raw_by_size = []
    for k in sizes:
        df_k = df[df["size"] == int(k)]
        d = {}
        for metric_name, g in df_k.groupby("metric"):
            g = g.sort_values("iter")
            d[metric_name] = g["value"].tolist()
        raw_by_size.append(d)
    return raw_by_size


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
                 num_models=35,
                 return_raw=False):
    
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
        
    if return_raw:
        return aver_metr, iter_vals
    
    return aver_metr

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
                        num_models=35,
                        return_raw=False):
    
    iter_vals_list = [{} for _ in range(len(sizes))]
    
    for i in tqdm(range(iter), desc=label):
        
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
    
    if return_raw:
        return metrics, iter_vals_list
    
    return metrics

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
                     models_bench_data,
                     return_raw=False):

    metrics = []
    raw_list = []

    if transp:
        if return_raw:
            metrics, iter_vals_list = average_metr_transp(
                fn, args, data, indexes_list, sizes, test_ind, datasets,
                ranks, add_data, models_bench_data, label,
                iter=test_iter, models_bench=models_bench, num_models=num_models,
                return_raw=True
            )
            metrics[0]['sizes'] = sizes
            return metrics, iter_vals_list
        else:
            metrics = average_metr_transp(
                fn, args, data, indexes_list, sizes, test_ind, datasets,
                ranks, add_data, models_bench_data, label,
                iter=test_iter, models_bench=models_bench, num_models=num_models
            )
            metrics[0]['sizes'] = sizes
            return metrics

    for n_clusters in tqdm(sizes, desc=label):
        if return_raw:
            metr, raw = average_metr(
                fn, args, data, indexes_list,
                n_clusters - 1 * (n_clusters == 110),
                test_ind, datasets, ranks, add_data, models_bench_data,
                iter=test_iter, models_bench=models_bench, num_models=num_models,
                return_raw=True
            )
            raw_list.append(raw)
        else:
            metr = average_metr(
                fn, args, data, indexes_list,
                n_clusters - 1 * (n_clusters == 110),
                test_ind, datasets, ranks, add_data, models_bench_data,
                iter=test_iter, models_bench=models_bench, num_models=num_models
            )

        metr['sizes'] = sizes
        metrics.append(metr)

    if return_raw:
        return metrics, raw_list
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
        
        # t=35
        # if recsys:
        #     t = 11
        
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
                     models_bench=False,
                     run_stats_tests=True,
                     stats_metrics=("MAE", "Spearman"),
                     stats_alpha=0.05,
                     stats_save_dir=None
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
    
    # expected = int(datasets.shape[0] * ratio)
    # check_train_inxs_list_duplicates(train_inxs_list, expected_len=expected)
    
    if test_datasets is not None:
        test_indxs = test_datasets
    if models_bench:
        test_indxs = np.arange(num_models)
    
    metr_list = []
    labels = []
    raw_results = {}
    
    time_save = re.sub(r'[-:\s.]', '_', str(datetime.now()))
    
    # -------------------------------------------------------------------------------------------------------------------------
    run_dir = None
    if save_results:
        run_dir = Path(save_res_path) / time_save
        run_dir.mkdir(parents=True, exist_ok=True)

        save_plot_path = str(run_dir)
        save_res_path  = str(run_dir)
        if stats_save_dir is None:
            stats_save_dir = str(run_dir)
    # -------------------------------------------------------------------------------------------------------------------------
    
    
    for (method, data, sizes, need_l, need_m, args, transp, models_in_data, label) in metods_data_list:
        
        # -------------------------------------------------------------------------------------------------------------------------
        base_dir = save_check_path if save_checpoints else os.path.join(save_history_path, label, time_save)
        method_dir = Path(base_dir) / label if save_checpoints else Path(base_dir)
        method_dir.mkdir(parents=True, exist_ok=True)

        metrics_pkl = method_dir / "metrics.pkl"
        raw_csv = method_dir / "raw_results.csv"
        summary_csv = method_dir / "summary_ci.csv"
        meta_json = method_dir / "meta.json"
        # -------------------------------------------------------------------------------------------------------------------------
        
        # if load_res and os.path.exists(os.path.join(save_check_path, f"{label}.pkl")):
        #     print(f"Loading {label} results...") 
            
        #     with open(os.path.join(save_check_path, f"{label}.pkl"), 'rb') as f:
        #         metr = pickle.load(f)
            
        #     print(f"{label} results successfully loaded")  
             
        #     labels.append(label) 
        #     metr_list.append(metr)
        #     continue
        
        
        if load_res and metrics_pkl.exists():
            print(f"Loading {label} results...")
            with open(metrics_pkl, "rb") as f:
                metr = pickle.load(f)

            if run_stats_tests and raw_csv.exists():
                raw_list = _load_raw_csv_to_structure(str(raw_csv), sizes=metr[0]["sizes"])
                raw_results[label] = {"sizes": metr[0]["sizes"], "raw": raw_list}

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
            lasso_feat = get_lasso_data(data=data,
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


        # metr = eval_method_mean(fn=method,
        #                         args=args,
        #                         transp=transp,
        #                         test_iter=test_iter,
        #                         data=data,
        #                         indexes_list=train_inxs_list,
        #                         sizes=sizes,
        #                         datasets=datasets,
        #                         ranks=ranks,
        #                         test_ind=test_indxs,
        #                         label=label,
        #                         add_data=[lasso_feat, entr_data, ranks_data],
        #                         models_bench=models_bench,
        #                         num_models=num_models,
        #                         models_bench_data=models_bench_data)
        
        metr, raw_list = eval_method_mean(fn=method,
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
                                models_bench_data=models_bench_data,
                                return_raw=True)
        
        
        raw_results[label] = {"sizes": sizes, "raw": raw_list}
        
        # if save_checpoints:
        #     with open(os.path.join(save_check_path, f"{label}.pkl"), 'wb') as f:
        #         pickle.dump(metr, f)
        
        with open(metrics_pkl, "wb") as f:
            pickle.dump(metr, f)

        _save_raw_csv(raw_list, sizes=sizes, out_csv=str(raw_csv))

        _save_summary_ci_csv(metr, out_csv=str(summary_csv))

        with open(meta_json, "w", encoding="utf-8") as f:
            json.dump({
                "label": label,
                "sizes": [int(x) for x in sizes],
                "test_iter": int(test_iter),
                "transp": bool(transp),
                "need_l": bool(need_l),
                "need_m": bool(need_m),
                "models_bench": bool(models_bench),
                "num_models": int(num_models),
                "time_save": time_save,
                "args": args,
            }, f, ensure_ascii=False, indent=2)
        
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
                 time_save=time_save,
                 raw_results=raw_results)
    
    
    # if run_stats_tests:
    #     if stats_save_dir is None:
    #         stats_save_dir = save_res_path

    #     if "MAE" in stats_metrics:
    #         run_friedman_holm_over_sizes(
    #             raw_results,
    #             metric="MAE",
    #             higher_is_better=False,
    #             alpha=stats_alpha,
    #             save_dir=stats_save_dir,
    #             print_tables=False
    #         )

    #     if "Spearman" in stats_metrics:
    #         run_friedman_holm_over_sizes(
    #             raw_results,
    #             metric="Spearman",
    #             higher_is_better=True,
    #             alpha=stats_alpha,
    #             save_dir=stats_save_dir,
    #             print_tables=False
    #         )




def _holm_adjust(pvals):

    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    p_sorted = pvals[order]

    adj_sorted = (m - np.arange(m)) * p_sorted

    adj_sorted = np.maximum.accumulate(adj_sorted)
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)

    p_adj = np.empty(m, dtype=float)
    p_adj[order] = adj_sorted
    
    return p_adj


def friedman_holm_table(raw_by_method,
                        metric,
                        higher_is_better,
                        alpha=0.05):

    labels = list(raw_by_method.keys())
    
    vals = [np.asarray(raw_by_method[l], dtype=float) for l in labels]
    
    n = min(len(v) for v in vals)
    
    vals = [v[:n] for v in vals]

    vals_for_friedman = vals if (not higher_is_better) else [(-v) for v in vals]

    stat, p = friedmanchisquare(*vals_for_friedman)

    ranks_mat = []
    
    for i in range(n):
        row = np.array([v[i] for v in vals], dtype=float)
        
        if higher_is_better:
            row = -row
            
        ranks = rankdata(row, method='average')
        ranks_mat.append(ranks)
        
    ranks_mat = np.vstack(ranks_mat)
    avg_ranks = ranks_mat.mean(axis=0)

    best_idx = int(np.argmin(avg_ranks))
    best_label = labels[best_idx]

    friedman_df = pd.DataFrame([{
        "metric": metric,
        "n_blocks": n,
        "n_methods": len(labels),
        "friedman_chi2": float(stat),
        "friedman_p": float(p),
        "best_by_avg_rank": best_label,
    }])

    best_vals = vals[best_idx]
    
    p_raw = []
    rows = []
    
    for j, lab in enumerate(labels):
        if j == best_idx:
            continue
        other_vals = vals[j]

        if higher_is_better:
            delta = best_vals - other_vals
            alt = "greater"
        else:
            delta = other_vals - best_vals
            alt = "greater"

        try:
            w_stat, pval = wilcoxon(delta, alternative=alt, zero_method="wilcox")
        except ValueError:
            w_stat, pval = np.nan, 1.0

        p_raw.append(pval)
        rows.append({
            "metric": metric,
            "best": best_label,
            "other": lab,
            "avg_rank_best": float(avg_ranks[best_idx]),
            "avg_rank_other": float(avg_ranks[j]),
            "median_delta(best_advantage)": float(np.median(delta)),
            "mean_delta(best_advantage)": float(np.mean(delta)),
            "wilcoxon_p_raw": float(pval),
        })

    if len(p_raw) > 0:
        p_adj = _holm_adjust(np.array(p_raw, dtype=float))
        
        for r, padj in zip(rows, p_adj):
            r["wilcoxon_p_holm"] = float(padj)
            r["reject_holm_alpha"] = bool(padj <= alpha)
            
    posthoc_df = pd.DataFrame(rows).sort_values("wilcoxon_p_raw")

    return friedman_df, posthoc_df


def run_friedman_holm_over_sizes(raw_results,
                                 metric="MAE",
                                 higher_is_better=False,
                                 alpha=0.05,
                                 save_dir=None,
                                 print_tables=True):
    
    friedman_rows = []
    holm_rows = []

    labels = list(raw_results.keys())
    
    size_sets = [set(raw_results[l]["sizes"]) for l in labels]
    common_sizes = sorted(set.intersection(*size_sets)) if len(size_sets) else []

    out = {}
    for k in common_sizes:
        raw_by_method = {}
        for lab in labels:
            sizes = list(raw_results[lab]["sizes"])
            j = sizes.index(k)
            raw_list = raw_results[lab]["raw"]
            raw_by_method[lab] = raw_list[j][metric]

        friedman_df, posthoc_df = friedman_holm_table(
            raw_by_method=raw_by_method,
            metric=metric,
            higher_is_better=higher_is_better,
            alpha=alpha
        )

        out[k] = {"friedman": friedman_df, "posthoc": posthoc_df}

        if print_tables:
            print("\n" + "=" * 90)
            print(f"[Friedman + Holm] metric={metric}, k={k}, alpha={alpha}")
            print(friedman_df.to_string(index=False))
            if len(posthoc_df) > 0:
                print("\nPost-hoc (best vs rest, Wilcoxon + Holm):")
                print(posthoc_df.to_string(index=False))

        if save_dir is not None:
            # os.makedirs(save_dir, exist_ok=True)
            # friedman_df.to_csv(os.path.join(save_dir, f"friedman_{metric}_k{k}.csv"), index=False)
            # posthoc_df.to_csv(os.path.join(save_dir, f"holm_{metric}_k{k}.csv"), index=False)
            
            for _, row in friedman_df.iterrows():
                friedman_rows.append({
                    "metric": metric,
                    "size": int(k),
                    **row.to_dict()
                })
            
            for _, row in posthoc_df.iterrows():
                holm_rows.append({
                    "metric": metric,
                    "size": int(k),
                    **row.to_dict()
                })

    if save_dir is not None:
        
        if len(friedman_rows) > 0:
            pd.DataFrame(friedman_rows).to_csv(
                os.path.join(save_dir, f"friedman_all.csv"),
                index=False
            )

        if len(holm_rows) > 0:
            pd.DataFrame(holm_rows).to_csv(
                os.path.join(save_dir, f"holm_all.csv"),
                index=False
            )
    
    return out