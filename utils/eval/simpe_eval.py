import os
import sys

if __name__ == '__main__':
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)


from utils.get_metrics import get_metrics
from utils.get_ranks import get_ranks
from tqdm import tqdm


def simple_eval(method_fn,
                sizes,
                data,
                datasets,
                label,
                full_ranks_data,
                ranks_all,
                **kwargs):
    
    results = []
    
    for sample_size in tqdm(sizes, desc=label):
        
        indx = method_fn(data, sample_size, **kwargs)

        repr_simple = datasets[indx].squeeze()

        ranks_simple = get_ranks(repr_simple, full_ranks_data)
        
        metr = get_metrics(ranks_all, ranks_simple)
        metr['repr'] = repr_simple
        metr['label'] = label
        
        results.append(metr)
    
    return results