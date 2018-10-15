from joblib import Parallel, delayed

from exps.train import run

Parallel(n_jobs=20, verbose=10)(delayed(run)(estimator, seed)
                                for estimator in ['factored']
                                for seed in range(10))
