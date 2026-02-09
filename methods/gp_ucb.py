import numpy as np
from sklearn.preprocessing import StandardScaler


def gp_ucb_indices(data,
                   sample_size,
                   **kwargs):

    defaults = {'init': 'point_center',
                'init_criteria': 'max',
                'scale_data': True,
                'length_scale': 1.0,
                'noise': 1e-6,
                'delta': 0.1,
                'random_state': None,
                'func': None}

    defaults.update(kwargs)
    
    (init, init_criteria, scale_data, length_scale,
     noise, delta, random_state, func) = defaults.values()

    rng = np.random.default_rng(random_state)

    X = np.asarray(data, dtype=float)
    if scale_data:
        X = StandardScaler().fit_transform(X)

    n = X.shape[0]
    all_idx = np.arange(n)
    chosen = []


    y_obs = []

    sigma2 = noise

    def rbf(a, b):
        sqd = np.sum(a**2, 1)[:, None] + np.sum(b**2, 1)[None, :] - 2*a@b.T
        return np.exp(-0.5 * sqd / length_scale**2)

    if init == 'point_center':
        center = X.mean(axis=0)
        dists = np.linalg.norm(X - center, axis=1)
        first = np.argmax(dists) if init_criteria == 'max' else np.argmin(dists)
    else:
        first = rng.choice(all_idx)

    chosen.append(first)
    y_obs.append(func(X[first]))

    for t in range(2, sample_size + 1):
        left = np.setdiff1d(all_idx, chosen, assume_unique=True)

        X_train = X[chosen]
        X_left  = X[left]

        K_tt = rbf(X_train, X_train) + sigma2*np.eye(len(chosen))
        K_ts = rbf(X_train, X_left)
        K_ss_diag = np.ones(len(left))

        K_inv = np.linalg.inv(K_tt)
        alpha = K_inv @ y_obs

        mu = (K_ts.T @ alpha)
        v = np.linalg.solve(K_tt, K_ts)
        sigma = np.sqrt(np.maximum(0.0, K_ss_diag - np.sum(K_ts * v, axis=0)))

        beta_t = 2 * np.log(n * t**2 * np.pi**2 / (6 * delta))
        ucb = mu + np.sqrt(beta_t) * sigma
        next_idx = left[np.argmax(ucb)]

        chosen.append(next_idx)
        y_obs.append(func(X[next_idx]))

    return np.array(chosen)
