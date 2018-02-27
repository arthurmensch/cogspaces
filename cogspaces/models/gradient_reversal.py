import torch
from torch import nn
from torch.nn import Linear
from torch.nn.functional import log_softmax, nll_loss



class LinearAutoEncoder(nn.Module):
    def __init__(self, in_features, out_features,
                 l1_penalty=0., l2_penalty=0.):
        super().__init__()
        self.linear = Linear(in_features, out_features, bias=False)
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, input):
        return self.linear(input)

    def reconstruct(self, projection):
        weight = self.linear.weight
        # gram = torch.matmul(weight.transpose(0, 1), weight)
        rec = torch.matmul(projection, weight)
        # rec, LU = torch.gesv(rec, gram)
        return rec

    def penalty(self):
        penalty = self.l2_penalty * .5 * torch.sum(self.linear.weight ** 2)
        penalty += self.l1_penalty * torch.sum(torch.abs(self.linear.weight))
        return penalty


class MultiClassifierHead(nn.Module):
    def __init__(self, in_features,
                 target_sizes, l1_penalty=0., l2_penalty=0.):
        super().__init__()
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.classifiers = {}
        for study, size in target_sizes.items():
            self.classifiers[study] = Linear(in_features, size)
            self.add_module('classifier_%s' % study, self.classifiers[study])
        self.softmax = nn.LogSoftmax(dim=1)
        self.reset_parameters()

    def reset_parameters(self):
        for classifier in self.classifiers.values():
            init.xavier_uniform(classifier.weight)
            classifier.bias.data.fill_(0.)

    def forward(self, input):
        preds = {}
        for study, sub_input in input.items():
            pred = self.softmax(self.classifiers[study](sub_input))
            preds[study] = pred
        return preds

    def penalty(self):
        penalty = 0
        for study, classifier in self.classifiers.items():
            penalty += self.l2_penalty * .5 * torch.sum(classifier.weight ** 2)
            penalty += self.l1_penalty * torch.sum(
                torch.abs(classifier.weight))
        return penalty


class MultiClassifierModule(nn.Module):
    def __init__(self, in_features, embedding_size,
                 target_sizes,
                 private_embedding_size=5,
                 dropout=0., input_dropout=0.,
                 l1_penalty=0., l2_penalty=0.):
        super().__init__()
        if embedding_size == 'auto':
            embedding_size = sum(target_sizes.values())
        self.embedder = LinearAutoEncoder(in_features,
                                          embedding_size,
                                          l1_penalty=l1_penalty,
                                          l2_penalty=l2_penalty)
        self.dropout = Dropout(dropout)
        self.input_dropout = Dropout(input_dropout)
        self.classifier_head = MultiClassifierHead(embedding_size +
                                                   private_embedding_size,
                                                   target_sizes,
                                                   l1_penalty=l1_penalty,
                                                   l2_penalty=l2_penalty)
        self.privates = {}
        for study in target_sizes:
            self.privates[study] = Linear(in_features, private_embedding_size)
            self.add_module('private_%s' % study, self.privates[study])

    def reset_parameters(self):
        self.embedder.reset_parameters()
        self.classifier_head.reset_parameters()
        for private in self.privates.values():
            private.reset_parameters()

    def forward(self, input):
        embeddings = {}
        for study, sub_input in input.items():
            sub_input = self.input_dropout(sub_input)
            private_embedding = self.privates[study](sub_input)
            embeddings[study] = self.dropout(torch.cat(
                (private_embedding, self.embedder(sub_input)), dim=1))
        return self.classifier_head(embeddings)

    def reconstruct(self, input):
        recs = {}
        for study, sub_input in input.items():
            recs[study] = self.embedder.reconstruct(self.embedder(sub_input))
        return recs

    def penalty(self):
        penalty = self.classifier_head.penalty() + self.embedder.penalty()
        return penalty


class MultiClassifierLoss(nn.Module):
    def __init__(self, size_average=False, study_weights=None):
        super().__init__()
        self.size_average = size_average
        self.study_weights = study_weights

    def forward(self, input, target):
        loss = 0
        for study in input:
            this_pred = input[study]
            this_target = target[study]
            weight = self.study_weights[study]
            loss += nll_loss(this_pred, this_target,
                             size_average=self.size_average) * weight
        return loss