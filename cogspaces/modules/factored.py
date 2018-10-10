import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from cogspaces.datasets import fetch_atlas_modl
from cogspaces.modules.linear import DropoutLinear


class Embedder(nn.Module):
    def __init__(self, in_features, latent_size, var_penalty,
                 dropout=0., adaptive=False,
                 init='normal'):
        super().__init__()

        self.init = init

        self.linear = DropoutLinear(in_features, latent_size,
                                    adaptive=adaptive,
                                    var_penalty=var_penalty,
                                    sparsify=False,
                                    init=init,
                                    level='additive' if adaptive else 'layer',
                                    p=dropout, bias=True)
        self.reset_parameters()

    def forward(self, input):
        return self.linear(input)

    def reset_parameters(self):
        if isinstance(self.init, str):
            if self.init == 'normal':
                self.linear.reset_parameters()
            elif self.init == 'orthogonal':
                nn.init.orthogonal_(self.linear.weight.data,
                                    gain=1 / math.sqrt(
                                        self.linear.weight.shape[1]))
            elif self.init == 'resting-state':
                assert self.linear.out_features == 128
                assert self.linear.in_features == 453
                dataset = fetch_atlas_modl()
                weight = np.load(dataset['loadings_128_gm'])
                self.linear.weight.data = torch.from_numpy(np.array(weight))
            else:
                raise ValueError('Wrong parameter for `init` %s' % self.init)
        elif isinstance(self.init, np.ndarray):
            self.linear.weight.data = torch.from_numpy(self.init)
        else:
            raise ValueError('Wrong parameter for `init` %s' % self.init)
        self.linear.reset_dropout()

    def penalty(self):
        return self.linear.penalty()


class LatentClassifier(nn.Module):
    def __init__(self, latent_size, target_size, var_penalty,
                 dropout=0., adaptive=False, batch_norm=True):
        super().__init__()

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(latent_size, affine=False, )
        self.linear = DropoutLinear(latent_size,
                                    target_size, bias=True, p=dropout,
                                    var_penalty=var_penalty,
                                    adaptive=adaptive,
                                    level='layer')

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
                 adaptive='embedding+classifier',
                 ):
        super().__init__()

        embedder_adaptive = 'embedding' in adaptive

        total_length = sum(list(lengths.values()))
        self.embedder = Embedder(in_features, latent_size,
                                 dropout=input_dropout,
                                 adaptive=embedder_adaptive,
                                 init=init,
                                 var_penalty=regularization / total_length)
        classifier_adaptive = 'classifier' in adaptive
        self.classifiers = {study: LatentClassifier(
            latent_size, target_size, dropout=latent_dropout,
            var_penalty=regularization / lengths[study],
            batch_norm=batch_norm,
            adaptive=classifier_adaptive, )
            for study, target_size in target_sizes.items()}
        for study, classifier in self.classifiers.items():
            self.add_module('classifier_%s' % study, classifier)

    def reset_parameters(self):
        self.embedder.reset_parameters()
        for classifier in self.classifiers.values():
            classifier.reset_parameters()

    def forward(self, inputs, logits=False):
        preds = {}
        for study, input in inputs.items():
            preds[study] = self.classifiers[study](self.embedder(input),
                                                   logits=logits)
        return preds

    def penalty(self, inputs):
        return (self.embedder.penalty()
                + sum(self.classifiers[study].penalty()
                      for study in inputs))