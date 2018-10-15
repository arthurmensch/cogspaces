from joblib import Parallel, delayed

from exps.train import run

Parallel(n_jobs=1, verbose=10)(delayed(run)(estimator, seed)
                                for estimator in ['logistic', 'estimator']
                                for seed in range(20))
