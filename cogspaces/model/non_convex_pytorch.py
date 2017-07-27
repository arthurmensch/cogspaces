import numpy as np
import tensorflow as tf
from keras import backend as K, Input
# Cut verbosity
# _stderr = sys.stderr
# null = open(os.devnull, 'wb')
# sys.stderr = null
from keras.callbacks import Callback, LearningRateScheduler
from keras.engine import Layer
from keras.engine import Model
from keras.layers import Dropout, Dense
from keras.optimizers import Adam, SGD, RMSprop
from keras.regularizers import l2
from modl.utils.math.enet import enet_projection
from numpy.linalg import svd
from sklearn.base import BaseEstimator
from sklearn.utils import gen_batches, check_random_state

MIN_FLOAT32 = np.finfo(np.float32).min


# sys.stderr = _stderr


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
        n_datasets = len(Xs)
        n_samples = sum(X.shape[0] for X in Xs)
        n_features = Xs[0].shape[1]

        # Data
        sizes = np.array([y.shape[1] for y in ys])
        limits = [0] + np.cumsum(sizes).tolist()
        n_targets = limits[-1]

        if self.n_components == 'auto':
            n_components = n_targets
        else:
            n_components = self.n_components

        self.slices_ = []
        for iter in range(n_datasets):
            self.slices_.append(np.array([limits[iter], limits[iter + 1]]))
        self.slices_ = tuple(self.slices_)

        if dataset_weights is None:
            dataset_weights = [1.] * n_datasets
        # dataset_weights = np.array(dataset_weights) * np.sqrt([X.shape[0] for X in Xs])
        # dataset_weights /= np.sum(dataset_weights) / n_datasets

        padded_ys = []
        masks = []
        Xs = [X.copy() for X in Xs]
        for y, this_slice in zip(ys, self.slices_):
            padded_y = np.zeros((y.shape[0], n_targets))
            mask = np.zeros((y.shape[0], n_targets), dtype='bool')
            mask[:, this_slice[0]:this_slice[1]] = 1
            padded_y[:, this_slice[0]:this_slice[1]] = y
            padded_ys.append(padded_y)
            masks.append(mask)
        if self.use_generator:
            our_generator = generator(Xs, padded_ys, masks,
                                      batch_size=self.batch_size,
                                      dataset_weights=dataset_weights,
                                      random_state=self.random_state)
            if self.batch_size is not None:
                steps_per_epoch = n_samples / self.batch_size
            else:
                steps_per_epoch = n_datasets
        else:
            Xs_cat = np.concatenate(Xs, axis=0)
            padded_ys_cat = np.concatenate(padded_ys, axis=0)
            masks_cat = np.concatenate(masks, axis=0)
            sample_weight = np.concatenate(
                [[dataset_weight * n_samples / X.shape[0] / n_datasets] * X.shape[0]
                 for dataset_weight, X in zip(dataset_weights, Xs)])
            if self.batch_size is None:
                batch_size = n_samples
            else:
                batch_size = self.batch_size
        # Model
        if self.intercept_init is not None:
            supervised_bias_initializer = \
                lambda shape: K.constant(self.intercept_init)
        else:
            supervised_bias_initializer = 'zeros'
        if self.coef_init is not None:
            U, S, VT = svd(self.coef_init, full_matrices=False)
            latent_init = U[:, :n_components]
            latent_init *= S[:n_components]
            supervised_init = VT[:n_components]
            if n_components > latent_init.shape[1]:
                latent_init = np.concatenate(
                    [latent_init, np.zeros((latent_init.shape[0],
                                            n_components -
                                            latent_init.shape[1]))],
                    axis=1)
                supervised_init = np.concatenate(
                    [supervised_init, np.zeros(
                        (n_components - supervised_init.shape[0],
                         supervised_init.shape[1]))],
                    axis=0)
            supervised_kernel_initializer = \
                lambda shape: K.constant(supervised_init)
            latent_kernel_initializer = \
                lambda shape: K.constant(latent_init)
        else:
            supervised_kernel_initializer = 'glorot_uniform'
            latent_kernel_initializer = 'glorot_uniform'
        data = Input(shape=(n_features,), name='data', dtype='float32')
        mask = Input(shape=(n_targets,), name='mask', dtype='bool')
        dropout_data = Dropout(rate=self.input_dropout_rate)(data)
        if n_components is not None:
            latent = Dense(n_components,
                           name='latent',
                           use_bias=False,
                           kernel_initializer=latent_kernel_initializer,
                           kernel_regularizer=l2(self.alpha))(dropout_data)
            latent = Dropout(rate=self.latent_dropout_rate)(latent)
        else:
            latent = dropout_data
        logits = Dense(n_targets,
                       use_bias=True,
                       name='supervised',
                       kernel_initializer=supervised_kernel_initializer,
                       bias_initializer=supervised_bias_initializer,
                       kernel_regularizer=l2(self.alpha))(latent)
        prediction = PartialSoftmax()([logits, mask])
        self.model_ = Model(inputs=[data, mask], outputs=prediction)
        if self.optimizer == 'adam':
            optimizer = Adam
        elif self.optimizer == 'sgd':
            optimizer = SGD
        elif self.optimizer == 'rmsprop':
            optimizer = RMSprop
        else:
            raise ValueError('Wrong optimizer')
        self.model_.compile(optimizer(self.step_size),
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
        schedule = lambda i: 1e-3 if i < self.max_iter // 2 else 1e-4
        scheduler = LearningRateScheduler(schedule)
        callbacks = [scheduler]

        if self.latent_sparsity is not None:
            callbacks.append(L1ProjCallback(radius=self.latent_sparsity))

        sess = tf.Session(
            config=tf.ConfigProto(
                device_count={'CPU': self.n_jobs},
                inter_op_parallelism_threads=self.n_jobs,
                intra_op_parallelism_threads=self.n_jobs,
                use_per_session_threads=True)
        )
        K.set_session(sess)

        if self.use_generator:
            self.model_.fit_generator(our_generator,
                                      steps_per_epoch=steps_per_epoch,
                                      verbose=2,
                                      epochs=self.max_iter,
                                      callbacks=callbacks)
        else:
            self.model_.fit([Xs_cat, masks_cat], padded_ys_cat,
                            sample_weight=sample_weight,
                            batch_size=batch_size,
                            verbose=2,
                            epochs=self.max_iter,
                            callbacks=callbacks)
        supervised, self.intercept_ = self.model_.get_layer(
            'supervised').get_weights()
        if self.n_components is not None:
            latent = self.model_.get_layer('latent').get_weights()[0]
            self.coef_ = latent.dot(supervised)
        else:
            self.coef_ = supervised

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


def generator(Xs, padded_ys, masks, dataset_weights, batch_size,
              random_state=None):
    if batch_size is None:
        batch_sizes = [X.shape[0] for X in Xs]
    else:
        batch_sizes = [batch_size] * len(Xs)
    batchers = [iter([]) for _ in Xs]
    while True:
        for i, (X, y, mask, dataset_weight, batcher, batch_size) in enumerate(
                zip(Xs, padded_ys, masks, dataset_weights,
                    batchers, batch_sizes)):
            try:
                batch = next(batcher)
            except StopIteration:
                permutation = random_state.permutation(X.shape[0])
                X[:] = X[permutation]
                y[:] = y[permutation]
                mask[:] = mask[permutation]
                batcher = gen_batches(X.shape[0], batch_size)
                batchers[i] = batcher
                batch = next(batcher)
            batch_dataset_weight = np.ones(batch.stop - batch.start) * dataset_weight
            yield [X[batch], mask[batch]], y[batch], batch_dataset_weight


class L1ProjCallback(Callback):
    def __init__(self, radius=1):
        Callback.__init__(self)
        self.radius = radius

    def on_batch_end(self, batch, logs=None):
        weights = self.model.get_layer('latent').get_weights()[0]
        proj_weights = np.empty_like(weights)
        for i in range(weights.shape[1]):
            enet_projection(weights[:, i], proj_weights[:, i], self.radius, 1)
        self.model.get_layer('latent').set_weights([proj_weights])

    def on_epoch_begin(self, epoch, logs=None):
        weights = self.model.get_layer('latent').get_weights()[0]
        print('sparsity', np.mean(weights == 0))


class PartialSoftmax(Layer):
    def __init__(self, **kwargs):
        super(PartialSoftmax, self).__init__(**kwargs)

    def build(self, input_shape):
        super(PartialSoftmax, self).build(input_shape)

    def call(self, inputs, **kwargs):
        logits, mask = inputs
        # Put logits to -inf for constraining probability to some support
        logits_min = tf.ones_like(logits) * MIN_FLOAT32
        logits = tf.where(mask, logits, logits_min)
        logits_max = K.max(logits, axis=1, keepdims=True)
        logits -= logits_max
        exp_logits = tf.exp(logits)
        sum_exp_logits = K.sum(exp_logits, axis=1, keepdims=True)
        return exp_logits / sum_exp_logits

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('A `PartialSoftmax` layer should be called '
                             'on a list of inputs.')
        input_shape = input_shape[0]
        return input_shape
