from joblib import Parallel, delayed

from exps.train import run

Parallel(n_jobs=1)(delayed(run)(estimator, seed)
                   for estimator in ['factored', 'logistic']
                   for seed in range(2))
