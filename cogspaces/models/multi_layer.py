from torch import nn


class MultiLayerClassifier(nn.Module):
    def __init__(self, in_features, target_sizes,
                 first_hidden_features=None,
                 second_hidden_features=None,
                 dropout=0):
        super().__init__()
        if first_hidden_features is not None:
            self.first_linear = nn.Linear(in_features, first_hidden_features,
                                          bias=False)
            classifier_features = first_hidden_features
        if second_hidden_features is not None:
            self.second_linear = nn.Linear(first_hidden_features,
                                           second_hidden_features,
                                           bias=False)
            classifier_features = second_hidden_features

        self.dropout = nn.Dropout(dropout)
        self.classifiers = {}
        for study, size in target_sizes.items():
            self.classifiers[study] = nn.Linear(classifier_features, size)
            self.add_module('classifiers_%s' % study,
                            self.classifiers[study])
        self.softmax = nn.LogSoftmax(dim=1)

    def reset_parameters(self):
        if hasattr(self, 'first_linear'):
            self.first_linear.reset_parameters()
        if hasattr(self, 'second_linear'):
            self.second_linear.reset_parameters()
        for classifier in self.classifiers:
            classifier.reset_parameters()

    def load_first_reduction(self, weight):
        self.first_linear.weight = weight

    def forward(self, Xs):
        preds = {}
        for study, data in Xs.items():
            reduced = data
            if hasattr(self, 'first_linear'):
                reduced = self.dropout(self.first_linear(reduced))
            if hasattr(self, 'second_linear'):
                reduced = self.dropout(self.second_linear(reduced))
            pred = self.softmax(self.classifiers[study](reduced))
            preds[study] = pred
        return preds
