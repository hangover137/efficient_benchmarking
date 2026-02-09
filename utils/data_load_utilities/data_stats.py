import pandas as pd
import numpy as np
from typing import List, Dict

def find_shape_of_datasets(models_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Creates a DataFrame with the shape (row and column count) of each dataset (pd.DataFrame) in the input dictionary.

    Args:
        - models_dict (Dict[str, pd.DataFrame]): Dictionary with dataset names as keys and pd.DataFrames as values.

    Returns:
        - DataFrame: A DataFrame where columns represent the datasets and rows represent the number of rows ('row')
                  and columns ('col') for each dataset.
    """
    shapes = {key: value.shape for key, value in models_dict.items()}
    return pd.DataFrame(shapes, index=['row', 'col'])
