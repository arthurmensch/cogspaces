import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state, gen_batches
from torch import nn

MIN_FLOAT32 = np.finfo(np.float32).min


class NonConvexEstimator(BaseEstimator):
    def __init__(self, alpha=1.,
                 n_components=25,
                 step_size=1e-3,
                 latent_dropout_rate=0.,
                 input_dropout_rate=0.,
                 coef_init=None,
                 optimizer='sgd',
                 intercept_init=None,
                 batch_size=256,
                 latent_sparsity=None,
                 use_generator=False,
                 random_state=None,
                 n_jobs=1,
                 max_iter=1000):
        self.alpha = alpha
        self.n_components = n_components
        self.max_iter = max_iter
        self.step_size = step_size
        self.latent_dropout_rate = latent_dropout_rate
        self.input_dropout_rate = input_dropout_rate
        self.coef_init = coef_init
        self.intercept_init = intercept_init
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.latent_sparsity = latent_sparsity
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.use_generator = use_generator

    def fit(self, Xs, ys, dataset_weights=None):
        self.random_state = check_random_state(self.random_state)
        batches_generators = [gen_batches(X.shape[0]) for X in Xs]
        n_iter = 0
        n_features = Xs[0].shape[1]

        embedding = nn.Sequential([nn.Dropout(),
                                   nn.Linear(in_features=n_features, out_features=30,
                                   bias=False)],
                                  nn.Dropout())
        for X, y in zip(Xs, ys):
            n_targets = y.max() + 1
            model = nn.Sequential([embedding,
                                   nn.Linear(30, n_targets)])
            criterion = nn.CrossEntropyLoss


        n_features = Xs[0].shape[1]

    def predict(self, Xs):
        n_datasets = len(Xs)
        X_cat = np.concatenate(Xs, axis=0)
        n_targets = self.model_.output_shape[1]
        sample_sizes = np.array([X.shape[0] for X in Xs])
        limits = [0] + np.cumsum(sample_sizes).tolist()
        sample_slices = []
        for i in range(n_datasets):
            sample_slices.append(np.array([limits[i], limits[i + 1]]))
        sample_slices = tuple(sample_slices)
        masks = []
        for X, this_slice in zip(Xs, self.slices_):
            mask = np.zeros((X.shape[0], n_targets), dtype='bool')
            mask[:, this_slice[0]:this_slice[1]] = 1
            masks.append(mask)
        mask_cat = np.concatenate(masks, axis=0)
        y_cat = self.model_.predict([X_cat, mask_cat])
        ys = []
        for X, sample_slice, target_slice in zip(Xs, sample_slices,
                                                 self.slices_):
            y = y_cat[sample_slice[0]:sample_slice[1],
                target_slice[0]:target_slice[1]]
            ys.append(y)
        ys = tuple(ys)
        return ys