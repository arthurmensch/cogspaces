import numpy as np
import tensorflow as tf
from cogspaces.model.convex import MIN_FLOAT32
from keras import backend as K, Input
# Cut verbosity
# _stderr = sys.stderr
# null = open(os.devnull, 'wb')
# sys.stderr = null
from keras.callbacks import Callback, LearningRateScheduler
from keras.engine import Model, Layer
from keras.layers import Dropout, Dense
from keras.optimizers import Adam, SGD, RMSprop
from keras.regularizers import l2
from modl.utils.math.enet import enet_projection
from numpy.linalg import svd
from sklearn.base import BaseEstimator


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

    def fit(self, Xs, ys, init=None):
        n_datasets = len(Xs)
        sizes = np.array([y.shape[1] for y in ys])
        limits = [0] + np.cumsum(sizes).tolist()
        n_targets = limits[-1]
        self.slices_ = []
        for iter in range(n_datasets):
            self.slices_.append(np.array([limits[iter], limits[iter + 1]]))
        self.slices_ = tuple(self.slices_)

        padded_ys = []
        masks = []
        for y, this_slice in zip(ys, self.slices_):
            padded_y = np.zeros((y.shape[0], n_targets))
            mask = np.zeros((y.shape[0], n_targets), dtype='bool')
            mask[:, this_slice[0]:this_slice[1]] = 1
            padded_y[:, this_slice[0]:this_slice[1]] = y
            padded_ys.append(padded_y)
            masks.append(mask)
        y_cat = np.concatenate(padded_ys, axis=0)
        X_cat = np.concatenate(Xs, axis=0)
        mask_cat = np.concatenate(masks, axis=0)
        n_samples, n_features = X_cat.shape
        sample_weight = np.concatenate([[1. / X.shape[0]] * X.shape[0]
                                        for X in Xs])
        sample_weight *= n_samples / n_datasets

        if self.batch_size is None:
            batch_size = n_samples
        else:
            batch_size = self.batch_size

        if self.intercept_init is not None:
            supervised_bias_initializer = \
                lambda shape: K.constant(self.intercept_init)
        else:
            supervised_bias_initializer = 'zeros'
        if self.coef_init is not None:
            U, S, VT = svd(self.coef_init, full_matrices=False)
            latent_init = U[:, :self.n_components]
            latent_init *= S[:self.n_components]
            supervised_init = VT[:self.n_components]
            if self.n_components > latent_init.shape[1]:
                latent_init = np.concatenate(
                    [latent_init, np.zeros((latent_init.shape[0],
                                            self.n_components -
                                            latent_init.shape[1]))],
                    axis=1)
                supervised_init = np.concatenate(
                    [supervised_init, np.zeros(
                        (self.n_components - supervised_init.shape[0],
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
        latent = Dense(self.n_components,
                       name='latent',
                       use_bias=False,
                       kernel_initializer=latent_kernel_initializer,
                       kernel_regularizer=l2(self.alpha))(dropout_data)
        latent = Dropout(rate=self.latent_dropout_rate)(latent)
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
        if self.latent_sparsity is not None:
            callbacks = [L1ProjCallback(radius=self.latent_sparsity)]
        else:
            callbacks = []

        def schedule(epoch):
            return self.step_size / (1 + epoch)
        callbacks.append(LearningRateScheduler(schedule))

        self.model_.fit([X_cat, mask_cat], y_cat,
                        sample_weight=sample_weight,
                        batch_size=batch_size, verbose=2,
                        epochs=self.max_iter,
                        callbacks=callbacks)

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
