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