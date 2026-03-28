import os

import numpy as np
import matplotlib.pyplot as plt


def plot_results(metr_list,
                 labels,
                 save_plot_path,
                 save_results,
                 save_res_path,
                 time_save,
                 raw_results=None):

    """
    Plot metric curves over subset size for all evaluated methods and print AUC
    summaries.
    
    This function expects the exact output structure produced by
    ``testing_pipeline`` / ``eval_method_mean``. For every metric it creates a
    figure showing the mean curve and the empirical 95% interval for each method.
    It also prints trapezoidal AUC summaries to stdout, optionally including the
    standard deviation of per-iteration AUC when raw results are available.
    
    Parameters
    ----------
    metr_list : list of list of dict
        One item per method. Each method entry is a list over subset sizes, and
        each size entry stores ``{'mean', 'q2.5', 'q97.5'}`` for every metric plus
        a shared ``'sizes'`` field.
    labels : sequence of str
        Method labels aligned with ``metr_list``.
    save_plot_path : str
        Directory where timestamped figures ``<metric>_all_<time_save>.png`` are
        written.
    save_results : bool
        If ``True``, also save a non-timestamped copy ``<metric>.png`` to
        ``save_res_path``.
    save_res_path : str
        Secondary directory used when ``save_results`` is enabled.
    time_save : str
        Timestamp suffix included in the main output filename.
    raw_results : dict, optional
        Raw per-iteration results in the format produced by ``testing_pipeline``.
        When provided, the function estimates the standard deviation of AUC across
        repetitions and prints it alongside the mean AUC.
    
    Returns
    -------
    None
        The function saves plots to disk and prints summary lines.
    
    Notes
    -----
    Methods whose label contains ``'Random'`` are plotted in black to highlight the
    baseline.
"""
    
    for metric_name in metr_list[0][0].keys():

        plt.figure(figsize=(8, 6))
        
        if metric_name == 'sizes':
            continue
        
        if metric_name == 'label':
            continue
        
        for metrics, label in zip(metr_list, labels):
            
            mean_vals = [m[metric_name]["mean"] for m in metrics]
            q025_vals = [m[metric_name]["q2.5"] for m in metrics]
            q975_vals = [m[metric_name]["q97.5"] for m in metrics]
            
            x = np.array(metrics[0]['sizes'])
            mean_auc = np.trapz(mean_vals, x)
            low_auc  = np.trapz(q025_vals, x)
            high_auc = np.trapz(q975_vals, x)
            
            auc_std_str = "AUC(std)=n/a"
            if raw_results is not None:
                rr = raw_results.get(label, None)
                raw_list = rr["raw"]
                if metric_name in raw_list[0]:
                    iters = len(raw_list[0][metric_name])
                    aucs = []
                    for it in range(iters):
                        y = np.array([raw_list[j][metric_name][it] for j in range(len(x))], dtype=float)
                        aucs.append(np.trapz(y, x))
                    aucs = np.asarray(aucs, dtype=float)
                    auc_std_str = f"AUC(std)={aucs.std():.2f}"
            
            print(
                f"{label:25s} | {metric_name:8s} | "
                f"AUC(mean)={mean_auc:.2f}, "
                f"{auc_std_str}, "
                f"AUC(q2.5)={low_auc:.4f}, "
                f"AUC(q97.5)={high_auc:.4f}, "
                f"delta={np.abs(high_auc-low_auc):.4f}"
            )
            
            if 'Random' in label:
                line_plot = plt.plot(metrics[0]['sizes'], mean_vals, label=label, c='black')
            else:
                line_plot = plt.plot(metrics[0]['sizes'], mean_vals, label=label)
            
            plt.fill_between(metrics[0]['sizes'], q025_vals, q975_vals, alpha=0.1)
            
            
            
            color_line = line_plot[0].get_color()
            plt.plot(metrics[0]['sizes'], q025_vals, c=color_line, ls='--', alpha=0.5)
            plt.plot(metrics[0]['sizes'], q975_vals, c=color_line, ls='--', alpha=0.5)
            
            
        
        plt.xlabel('Sample Size')
        plt.ylabel(metric_name)
        plt.grid()
        plt.tight_layout()
        plt.legend(fontsize=12)
        plt.xticks(metrics[0]['sizes'])
        plt.savefig(os.path.join(save_plot_path, f"{metric_name}_all_{time_save}.png"))
        
        if save_results:
            plt.savefig(os.path.join(save_res_path, f"{metric_name}.png"), dpi=300)
        
        plt.plot()