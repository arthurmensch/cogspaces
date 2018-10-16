from joblib import Parallel, delayed
from sklearn.utils import check_random_state

from exps.train import run

seeds = check_random_state(42).randint(0, 100000, size=20).tolist()

Parallel(n_jobs=40, verbose=10)(delayed(run)(estimator, seed)
                                for estimator in ['multi_study', 'logistic']
                                for seed in seeds)
