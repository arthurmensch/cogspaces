import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from cogspaces.datasets import fetch_atlas_modl
from cogspaces.modules.linear import DropoutLinear


class LatentClassifier(nn.Module):
    def __init__(self, latent_size, target_size, var_penalty,
                 dropout=0., adaptive=False, batch_norm=True):
        """
        One third-layer classification head.

        Simply combines batch-norm -> DropoutLinear -> softmax.

        Parameters
        ----------
        latent_size : int
            Size of the latent space.

        target_size : int
            Number of targets for the classifier.

        var_penalty : float
            Penalty to apply for variational latent_dropout

        dropout : float, [0, 1]
            Dropout rate to apply at the input

        adaptive : bool
            Use adaptive latent_dropout rate

        batch_norm : bool
            Use batch normalization at the input
        """
        super().__init__()

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(latent_size, affine=False, )
        self.linear = DropoutLinear(latent_size,
                                    target_size, bias=True, p=dropout,
                                    var_penalty=var_penalty,
                                    adaptive=adaptive, level='layer')

    def forward(self, input, logits=False):
        if hasattr(self, 'batch_norm'):
            if not self.training or len(input) > 1:
                input = self.batch_norm(input)
        logits_ = self.linear(input)
        if logits:
            return logits_
        else:
            return F.log_softmax(logits_, dim=1)

    def reset_parameters(self):
        self.linear.reset_parameters()
        self.batch_norm.reset_parameters()

    def penalty(self):
        return self.linear.penalty()

    def get_dropout(self):
        return self.linear.get_dropout()


class VarMultiStudyModule(nn.Module):
    def __init__(self, in_features,
                 latent_size,
                 target_sizes,
                 lengths,
                 input_dropout=0.,
                 regularization=1.,
                 latent_dropout=0.,
                 init='orthogonal',
                 batch_norm=True,
                 adaptive=False,
                 ):
        """
        Second and third-layer of the models.

        Parameters
        ----------
        in_features : int
            Size of the input before the second layer
            (number of resting-state loadings).

        latent_size : int
            Size of the latent dimension, in between the second and third layer.

        target_sizes: Dict[str, int]
            For each study, number of contrasts to predict

        lengths: Dict[str, int]
            Length of each study (for variational regularization)

        input_dropout: float, [0, 1]

        regularization: float, default=1
            Regularization to apply for variational latent_dropout

        latent_dropout: float, default=1
            Dropout rate to apply in between the second and third layer.

        init: str, {'normal', 'orthogonal', 'resting-state'}
            How to initialize the second layer. If 'resting-state',
            then it must be in_features = 453 and latent_size = 128

        batch_norm: bool,
            Batch norm between the second and third layer

        adaptive: bool,
            Dropout rate should be adaptive
        """
        super().__init__()

        total_length = sum(list(lengths.values()))
        self.embedder = DropoutLinear(
            in_features, latent_size, adaptive=adaptive,
            var_penalty=regularization / total_length, level='layer',
            p=input_dropout, bias=True)
        self.classifiers = {study: LatentClassifier(
            latent_size, target_size, dropout=latent_dropout,
            var_penalty=regularization / lengths[study],
            batch_norm=batch_norm, adaptive=adaptive, )
            for study, target_size in target_sizes.items()}
        for study, classifier in self.classifiers.items():
            self.add_module('classifier_%s' % study, classifier)
        self.init = init

    def reset_parameters(self):
        self.embedder.weight.data = self.get_embedder_init()
        nn.init.zeros_(self.embedder.bias.data)
        self.embedder.reset_dropout()
        for classifier in self.classifiers.values():
            classifier.reset_parameters()

    def get_embedder_init(self):
        weight = torch.empty_like(self.embedder.weight.data)
        gain = 1. / math.sqrt(weight.shape[1])
        if self.init == 'normal':
            self.weight.data.uniform_(-gain, gain)
        elif self.init == 'orthogonal':
            nn.init.orthogonal_(weight, gain=gain)
        elif self.init == 'resting-state':
            dataset = fetch_atlas_modl()
            weight = np.load(dataset['loadings_128_gm'])
            weight = torch.from_numpy(np.array(weight))
        return weight

    def forward(self, inputs, logits=False):
        preds = {}
        for study, input in inputs.items():
            preds[study] = self.classifiers[study](self.embedder(input),
                                                   logits=logits)
        return preds

    def penalty(self, studies):
        """
        Return the variational penalty of the model.

        Parameters
        ----------
        studies: Iterable[str],
            Studies to consider when computing the penalty

        Returns
        -------
        penalty: torch.tensor,
            Scalar penalty
        """
        return (self.embedder.penalty()
                + sum(self.classifiers[study].penalty()
                      for study in studies))

    def get_dropout(self):
        return (self.embedder.get_dropout(),
                {study: classifier.get_dropout() for study, classifier in
                 self.classifiers.items()})
