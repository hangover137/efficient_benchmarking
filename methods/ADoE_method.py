import numpy as np
import torch
import gpytorch

if __name__ == '__main__':
    
    from model_training.auto_training import ExactGPModel, train_model
    
else:
    
    from methods.model_training.auto_training import ExactGPModel, train_model

#------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def get_imse_L2(model,
                curr_sample,
                new_x,
                dv = 1):

    model.eval()
    
    test_x = torch.cat([new_x, curr_sample], dim=0)
    
    post = model(test_x)
    
    post_K_matr = post.covariance_matrix
    
    sigma_2_x_X_n_1 = post_K_matr[0, 0]
    
    K_N_1 = post_K_matr[0, 1:]
    
    K_2_N_1 = K_N_1**2
    
    imse_L2 = torch.mean(K_2_N_1) / (sigma_2_x_X_n_1 + 1e-10)

    return -sigma_2_x_X_n_1.item()
    
#------------------------------------------------------------------------------------------------------------

def adaptive_selection(train_x,
                       n_iter, # sample size
                       n_init=1,
                       random_state=42,
                       **kwargs):
    
    train_y = train_x[:, -1]
    train_x = train_x[:, :-1]
    
    np.random.seed(random_state)

    all_ind = list(range(len(train_x)))
    init_ind = np.random.choice(all_ind, n_init, replace=False)
    
    n_iter -= n_init
    
    chosen_ind = list(init_ind)
    new_ind = list(set(all_ind) - set(chosen_ind))

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
    model = ExactGPModel(train_x[chosen_ind], train_y[chosen_ind], likelihood)
    
    model, likelihood = train_model(train_x[chosen_ind], train_y[chosen_ind],
                                    model, likelihood, iter=500)


    for step in range(n_iter):
        
        best_crit = -1e10
        best_idx = None
        
        
        for idx in new_ind:
            
            cand_x = train_x[idx].unsqueeze(0)
            
            crit = get_imse_L2(model, train_x.clone(), cand_x)
            
            if crit > best_crit:
                best_crit = crit
                best_idx = idx

        chosen_ind.append(best_idx)
        
        new_ind.remove(best_idx)

        model = ExactGPModel(train_x[chosen_ind], train_y[chosen_ind],likelihood)
        
        model, likelihood = train_model(train_x[chosen_ind],train_y[chosen_ind], 
                                        model, likelihood, iter=500)

    return np.array(chosen_ind)