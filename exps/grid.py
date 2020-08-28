from joblib import Parallel, delayed
from sklearn.utils import check_random_state

from exps.train import run

seeds = check_random_state(42).randint(0, 100000, size=20).tolist()

# Takes 15 minutes on a 40 CPU computer
Parallel(n_jobs=40, verbose=10)(delayed(run)(estimator, seed, split_by_task=True)
                                for estimator in ['multi_study', 'logistic']
                                for seed in seeds)

# Takes 20h on a 40 CPU computer (1 hour per split) -- ensemble is more costly
# for seed in seeds:
#     run(estimator='ensemble', seed=seed, plot=False, n_jobs=40)
