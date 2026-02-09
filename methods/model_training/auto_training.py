import torch
import gpytorch

from sklearn.model_selection import train_test_split


class ExactGPModel(gpytorch.models.ExactGP):
    
    def __init__(self, train_x, train_y, likelihood):
        
        super().__init__(train_x, train_y, likelihood)
        
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.RBFKernel()


    def forward(self, x):
        
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

#------------------------------------------------------------------------------------------------------------

def get_kernel(kernel_type):
    if kernel_type == 'RBF':
        return gpytorch.kernels.RBFKernel()
    elif kernel_type == 'Matern_1.5':
        return gpytorch.kernels.MaternKernel(nu=1.5)
    elif kernel_type == 'Matern_2.5':
        return gpytorch.kernels.MaternKernel(nu=2.5)
    elif kernel_type == 'RationalQuadratic':
        return gpytorch.kernels.RationalQuadraticKernel()
    elif kernel_type == 'Periodic':
        return gpytorch.kernels.PeriodicKernel()
    elif kernel_type == 'Linear':
        return gpytorch.kernels.LinearKernel()
    elif kernel_type == 'RBF+Matern_1.5':
        return gpytorch.kernels.RBFKernel() + gpytorch.kernels.MaternKernel(nu=1.5)
    elif kernel_type == 'Matern_1.5+Periodic':
        return gpytorch.kernels.MaternKernel(nu=1.5) + gpytorch.kernels.PeriodicKernel()
    elif kernel_type == 'RBF+RationalQuadratic':
        return gpytorch.kernels.RBFKernel() + gpytorch.kernels.RationalQuadraticKernel()

#------------------------------------------------------------------------------------------------------------

def train_model_es(train_x,
                   train_y,
                   val_x,
                   val_y,
                   model,
                   likelihood, 
                    max_iter=300,
                    lr=0.1,
                    impr_part=0.01,
                    patience=5):
    
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    best_val_loss = None
    epochs_ni = 0
    best_epoch = 0
    
    for epoch in range(max_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            val_output = model(val_x)
            val_loss = -mll(val_output, val_y).item()
        model.train()
        likelihood.train()
        
        if best_val_loss is None or val_loss < best_val_loss:
            
            if best_val_loss is not None:
                r_imp = (best_val_loss - val_loss) / (abs(best_val_loss) + 1e-8)
            else:
                r_imp = float('inf')
                
            best_val_loss = val_loss
            
            best_epoch = epoch
            
            epochs_ni = 0
            
        else:
            r_imp = (best_val_loss - val_loss) / (abs(best_val_loss) + 1e-8)
            
            if r_imp < impr_part:
                
                epochs_ni += 1
            else:
                epochs_ni = 0
        
        if epochs_ni >= patience:
            break

    return model, likelihood, best_val_loss, epoch + 1

#------------------------------------------------------------------------------------------------------------

def automated_model_training(X,
                             y,
                             max_iter=3000,
                             lr=0.1,
                             impr_part=0.01,
                             patience=5,
                             random_state=42):
    
    
    train_x_np, val_x_np, train_y_np, val_y_np = train_test_split(X,
                                                                  y,
                                                                  test_size=0.2,
                                                                  random_state=random_state)
    
    train_x = torch.from_numpy(train_x_np).float()
    train_y = torch.from_numpy(train_y_np).float()
    val_x = torch.from_numpy(val_x_np).float()
    val_y = torch.from_numpy(val_y_np).float()
    
    best_overall_loss = float('inf')
    best_h = None
    best_epoch = None

    kernel_grid = ['RBF', 'Matern_1.5', 'Periodic']
    
    l_grid = [0.1, 1.0, 10.0]
    n_grid = [0.1, 1.0, 10.0]
    
    with gpytorch.settings.cholesky_jitter(1e-6):
    
        for k in kernel_grid:
            
            for l in l_grid:
                
                for n in n_grid:
                    
                    kernel = get_kernel(k)
                    kernel.lengthscale = torch.tensor(l)
                    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise=n) 
                    model = ExactGPModel(train_x, train_y, likelihood)
                    model.covar_module = gpytorch.kernels.ScaleKernel(kernel)
                    
                    model, likelihood, val_loss, used_epochs = train_model_es(
                        train_x, train_y, val_x, val_y, model, likelihood,
                        max_iter=max_iter, lr=lr, impr_part=impr_part, patience=patience
                    )
                    
                    if val_loss < best_overall_loss:
                        
                        best_overall_loss = val_loss
                        
                        best_h = {
                            'k': k,
                            'l': l,
                            'n': n
                        }
                        
                        best_epoch = used_epochs

        X_full_np = X
        y_full_np = y
        
        X_full = torch.from_numpy(X_full_np).float()
        y_full = torch.from_numpy(y_full_np).float()
        
        best_k = get_kernel(best_h['k'])
        best_k.lengthscale = torch.tensor(best_h['l'])
        full_l = gpytorch.likelihoods.GaussianLikelihood(noise=best_h['n'])
        full_m = ExactGPModel(X_full, y_full, full_l)
        full_m.covar_module = gpytorch.kernels.ScaleKernel(best_k)
        
        full_m.train()
        full_l.train()
        optimizer = torch.optim.Adam(full_m.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(full_l, full_m)
        
        for epoch in range(best_epoch):
            optimizer.zero_grad()
            output = full_m(X_full)
            loss = -mll(output, y_full)
            loss.backward()
            optimizer.step()
    
    return full_m, full_l, best_h

#------------------------------------------------------------------------------------------------------------

def train_model(train_x,
                train_y,
                model,
                likelihood,
                iter=500,
                lr=0.1):
    
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(iter):
        
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    return model, likelihood