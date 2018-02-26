import torch
from torch import nn
from torch.nn import Linear, init
from torch.nn.functional import log_softmax, nll_loss


class GradientReversal(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input

    def backward(self, grad):
        return - grad


class GradientReversalClassifier(nn.Module):
    def __init__(self, in_features, embedding_size,
                 target_sizes):
        super().__init__()
        self.embedders = {}
        self.classifiers = {}
        for study, size in target_sizes.items():
            self.embedders[study] = Linear(in_features, embedding_size)
            self.classifiers[study] = Linear(embedding_size, size)
            self.add_module('embedder_%s' % study, self.embedders[study])
            self.add_module('classifier_%s' % study, self.classifiers[study])
        self.study_classifier = Linear(embedding_size,
                                       len(target_sizes))
        self.reset_parameters()

    def reset_parameters(self):
        for classifier in self.classifiers.values():
            init.xavier_uniform(classifier.weight)
            classifier.bias.data.fill_(0.)
        for embedder in self.embedders.values():
            init.xavier_uniform(embedder.weight)
        init.xavier_uniform(self.study_classifier.weight)
        self.study_classifier.bias.fill_(0.)

    def forward(self, input):
        preds = {}
        study_preds = {}
        for study, sub_input in input.items():
            embedding = self.embedders[study](self.input_dropout(sub_input))
            embedding = self.dropout(embedding)
            study_preds[study] = log_softmax(self.study_classifier(embedding))
            preds[study] = log_softmax(self.classifiers[study](embedding))
        return torch.cat((study_preds[:, :, None], preds[:, :, None]), dim=2)


class GradientReversalLoss(nn.Module):
    def __init__(self, size_average=False, study_weight=None):
        super().__init__()
        self.size_average = size_average
        self.study_weight = study_weight

    def forward(self, inputs, targets):
        loss = 0
        study_preds, preds = inputs
        study_targets, targets = targets

        for study in inputs:
            pred = preds[study]
            target = targets[study]

            for i in range(2):
                loss += nll_loss(pred[:, :, i], target[:, i],
                                 size_average=self.size_average)
        return loss

