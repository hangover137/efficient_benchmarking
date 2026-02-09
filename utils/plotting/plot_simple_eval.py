import os

import matplotlib.pyplot as plt


def plot_simple_eval(methods_results: dict,
                     sizes,
                     plot_size=8,
                     save_res=False,
                     save_path=os.path.join('results', 'simple_eval_res')):
    
    for metric_name in methods_results[0][0].keys():
        
        if metric_name == 'repr': continue
        if metric_name == 'label': continue
        
        plt.figure(figsize=(plot_size, plot_size))

        
        for method_res in methods_results:
            
            if method_res[0]['label'] == 'rand':
                plt.plot(sizes, [m[metric_name] for m in method_res], label=method_res[0]['label'], c='black')
                continue
            
            plt.plot(sizes, [m[metric_name] for m in method_res], label=method_res[0]['label'], linestyle='--')
        
        
        plt.xlabel('Sample size')
        plt.ylabel(metric_name)
        plt.tight_layout()
        plt.ylabel(metric_name)
        plt.grid()
        plt.legend()
        
        if save_res:
            plt.savefig(os.path.join(save_path, f"{metric_name}_all.png"))
        
        plt.plot()
        
        