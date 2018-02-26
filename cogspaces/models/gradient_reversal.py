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
                 target_sizes,
                 input_dropout=0.,
                 share_embedding=True,
                 dropout=0.):
        super().__init__()
        self.share_embedding = share_embedding
        if self.share_embedding:
            self.embedder = Linear(in_features, embedding_size, bias=False)
        else:
            self.embedders = {}
        self.classifiers = {}
        for study, size in target_sizes.items():
            if not self.share_embedding:
                self.embedders[study] = Linear(in_features, embedding_size,
                                               bias=False)
                self.add_module('embedder_%s' % study, self.embedders[study])
            self.classifiers[study] = Linear(embedding_size, size)
            self.add_module('classifier_%s' % study, self.classifiers[study])
        self.study_classifier = Linear(embedding_size,
                                       len(target_sizes))
        self.dropout = nn.Dropout(p=dropout)
        self.input_dropout = nn.Dropout(p=input_dropout)

    def reset_parameters(self):
        for classifier in self.classifiers.values():
            init.xavier_uniform(classifier.weight)
            classifier.bias.data.fill_(0.)
        if not self.share_embedding:
            for embedder in self.embedders.values():
                init.xavier_uniform(embedder.weight)
        else:
            init.xavier_uniform(self.embedder.weight)
        init.xavier_uniform(self.study_classifier.weight)
        self.study_classifier.bias.data.fill_(0.)

    def forward(self, input):
        preds = {}
        for study, sub_input in input.items():
            sub_input = self.input_dropout(sub_input)
            if self.share_embedding:
                embedding = self.embedder(sub_input)
            else:
                embedding = self.embedders[study](sub_input)
            embedding = self.dropout(embedding)
            study_pred = log_softmax(self.study_classifier(embedding), dim=1)
            pred = log_softmax(self.classifiers[study](embedding), dim=1)
            preds[study] = study_pred, pred
        return preds


class GradientReversalLoss(nn.Module):
    def __init__(self, size_average=False, study_weights=None):
        super().__init__()
        self.size_average = size_average
        self.study_weights = study_weights

    def forward(self, inputs, targets):
        loss = 0
        for study in inputs:
            pred = inputs[study]
            target = targets[study]

            for i in range(2):
                loss += nll_loss(pred[i], target[:, i],
                                 size_average=self.size_average)
                # if i == 0:
                #     print(pred[i], target[:, i])
            loss *= self.study_weights[study]
        return loss
