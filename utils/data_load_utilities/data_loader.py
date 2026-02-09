from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
import os
import json
from tqdm import tqdm
from aeon.datasets import load_classification

import pandas as pd
import numpy as np
from typing import List, Dict, Sequence, Tuple

PAGE_LINK = "https://timeseriesclassification.com/results/PublishedResults/"

def get_html_page_and_prepare_soup(page_link: str) -> BeautifulSoup:
    """
    Fetches an HTML page from a given URL and parses it into a BeautifulSoup object.

    Args:
        - page_link (str): URL of the page to be fetched.

    Returns:
        - BeautifulSoup: Parsed HTML content of the page.
    """
    response = requests.get(page_link)
    html = response.content
    soup = BeautifulSoup(html, "html.parser")
    return soup

def get_content_list_from_html(soup: BeautifulSoup, tag_name: str) -> List[str]:
    """
    Extracts text content from all HTML elements with the given tag and returns it as a list of strings.

    Args:
        - soup (BeautifulSoup): Parsed HTML content (soup object).
        - tag_name (str): The name of the HTML tag to search for.

    Returns:
        - List[str]: A list of text content from each found HTML element.
    """
    return [elem.get_text().strip() for elem in soup.find_all(tag_name)]

def load_model_results(paper_list: List[str], metrics_dir_name: str, 
                       need_download: bool = True) -> Dict[str, Dict[str, pd.DataFrame]]:
    '''
    Takes paper_list, downloads it from the 'timeseriesclassification.com' website if need_download is True
    to metrics_dir_name/paper_name directories. Then opens all model's results csv files as pd.DataFrames.

    Args:
        - paper_list: List[str] - The list with titles of papers with "name/" format
        - need_download - The bool specifies whether the models should be loaded from the website or only loaded into Python.

    Returns:
        - Dictionary[str, Dict[str, pd.DataFrame]]:
            * keys: string of paper name
            * values: a dictionary of models in pd.DataFrame format 
    '''
    paper_models_dict = {}
    
    for paper in paper_list:
        if not os.path.isdir(metrics_dir_name + "/" + paper[:-1]):
            os.mkdir(metrics_dir_name + "/" + paper[:-1])
            
        print(f"Parsing {paper[:-1]} models...\n")
        page_link_paper = PAGE_LINK + paper
    
        soup_i = get_html_page_and_prepare_soup(page_link_paper)
        models_list = get_content_list_from_html(soup_i, 'a')[1:]
    
        pd_models_dict = {}
        for model_name in models_list:
            if need_download:
                file_response = requests.get(page_link_paper + model_name, stream=True)
        
                with open(metrics_dir_name + '/' + paper + model_name, "wb") as handle:
                    for data in tqdm(file_response.iter_content()):
                        handle.write(data)
            pd_models_dict.update({model_name.split('_')[0]:pd.read_csv(metrics_dir_name + '/' + paper + model_name)})
        paper_models_dict.update({paper[:-1]:pd_models_dict})
    return paper_models_dict

def get_size_of_file_in_mb(json_filename: str) -> float:
    """
    Calculates the size of a file in megabytes (MB).
    Args:
        json_filename (str): Path to the file.
    Returns:
        float: Size of the file in MB, rounded to two decimal places.
    """
    file_size_bytes = os.path.getsize(json_filename)
    file_size_mb = file_size_bytes / (1024 * 1024)
    return round(file_size_mb, 2)

def process_datasets(dataset_lists: Sequence[str], dataset_dir_name: str) -> Tuple[Dict[str, float], List[int]]:
    """
    Processes a list of datasets by loading, serializing, and saving them as JSON files.
    Args:
        dataset_lists (list of str): List of dataset names to process.
        dataset_dir_name (str): Directory where JSON files will be saved.
    Returns:
        tuple:
            - file_sizes_mb (dict): Dictionary mapping dataset names to their JSON file sizes in MB.
            - problematic_datasets (list): List of dataset indices that encountered errors during processing.
    """
    file_sizes_mb = {}
    problematic_datasets = []
    
    for idx, dataset in enumerate(dataset_lists):
        json_filename = os.path.join(dataset_dir_name, f"{dataset}.json")
    
        if os.path.isfile(json_filename):
            tqdm.write(f"Skipping '{dataset}': JSON file already exists.")
            continue
            
        try:
            X, y, meta_data = load_classification(dataset, return_metadata=True)
        except Exception as e:
            tqdm.write(f"Error loading dataset '{dataset}': {e}")
            problematic_datasets.append(idx)
            continue
        
        data_dict = {
            'X': X.tolist() if isinstance(X, np.ndarray) else X,
            'y': y.tolist() if isinstance(X, np.ndarray) else y,
            'meta_data': meta_data
        }
        
        try:
            with open(json_filename, 'w') as json_file:
                json.dump(data_dict, json_file)

            file_sizes_mb[dataset] = get_size_of_file_in_mb(json_filename)
        
            tqdm.write(f"Saved '{dataset}.json' successfully. Size: {file_sizes_mb[dataset]} MB.")
        except Exception as e:
            tqdm.write(f"Error saving JSON for dataset '{dataset}' at index {idx}: {e}")
            problematic_datasets.append(idx)
            if os.path.isfile(json_filename):
                os.remove(json_filename)
            raise e

    print("\n=== Processing Summary ===")
    print(f"Total datasets to process: {len(dataset_lists)}")
    print(f"Successfully saved JSON files for {len(file_sizes_mb)} datasets.")
    print(f"Number of problematic datasets: {len(problematic_datasets)}")
    return file_sizes_mb, problematic_datasets

def load_datasets_from_json(dataset_lists: Sequence[str], dataset_dir_name: str) -> Dict[str, Tuple]:
    """
    Loads datasets from JSON files, deserializes the data, and converts lists back to NumPy arrays.
    Args:
        dataset_lists (list of str): List of dataset names to load.
        DATASET_DIR_NAME (str): Directory where the JSON files are stored.
    Returns:
        loaded_datasets (dict): Dictionary mapping dataset names to tuples of (X, y, meta_data).
    """
    loaded_datasets = {}
    for idx, dataset in enumerate(tqdm(dataset_lists)):
        json_filename = os.path.join(dataset_dir_name, f"{dataset}.json")
        
        with open(json_filename, 'r') as json_file:
            data_dict = json.load(json_file)
            
        X = np.array(data_dict['X'])
        y = np.array(data_dict['y'])
        meta_data = data_dict['meta_data']
        loaded_datasets[dataset] = (X, y, meta_data)
    return loaded_datasets