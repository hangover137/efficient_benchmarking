import os
import sys

import pandas as pd

from pathlib import Path

import warnings
warnings.simplefilter('ignore')

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from utils.data_load_utilities.data_loader import load_model_results

def get_global_const(DATA_DIR_NAME = 'data' + '/' + 'datasets_metrics',
                     DATASET_DIR_NAME = 'data' + '/' + 'datasets_metrics' + '/' + 'datasets',
                     METRICS_DIR_NAME = 'data' + '/' + 'datasets_metrics' + '/' + 'metrics',
                     paper_list=['Bakeoff2017/', 'Bakeoff2021/', 'Bakeoff2023/', 'HIVE-COTEV2/'],
                     scores_type='Bakeoff2023'):
    
    DATA_DIR_NAME = Path(DATA_DIR_NAME)
    DATASET_DIR_NAME = Path(DATASET_DIR_NAME)
    METRICS_DIR_NAME = Path(METRICS_DIR_NAME)
    
    paper_models_dict = load_model_results(paper_list, str(METRICS_DIR_NAME), need_download=False)
    scores = paper_models_dict[scores_type]
    
    models = list(scores.keys())
    datasets = (list(scores['cBOSS']["folds:"]))
    
    return scores, datasets, models



if __name__ == '__main__':

    print(get_global_const())