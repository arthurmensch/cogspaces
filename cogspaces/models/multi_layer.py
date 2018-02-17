from torch import nn
from torch.nn import Linear, Dropout


class MultiClassifierHead(nn.Module):
    def __init__(self, in_features,
                 target_sizes):
        super().__init__()
        self.classifiers = {}
        for study, size in target_sizes.items():
            self.classifiers[study] = Linear(in_features, size)
            self.add_module('classifiers_%s' % study, self.classifiers[study])
        self.softmax = nn.LogSoftmax(dim=1)

    def reset_parameters(self):
        for classifier in self.classifiers.values():
            classifier.reset_parameters()

    def forward(self, input):
        preds = {}
        for study, sub_input in input.items():
            pred = self.softmax(self.classifiers[study](sub_input))
            preds[study] = pred
        return preds


class MultiClassifier(nn.Module):
    def __init__(self, in_features, embedding_size,
                 target_sizes, dropout=0, input_dropout=0):
        super().__init__()
        self.input_dropout = Dropout(input_dropout)
        self.embedder = Linear(in_features, embedding_size)
        self.dropout = Dropout(dropout)
        self.classifier_head = MultiClassifierHead(embedding_size,
                                                   target_sizes)

    def reset_parameters(self):
        self.embedder.reset_parameters()
        self.classifier_head.reset_parameters()

    def forward(self, input):
        embeddings = {}
        for study, sub_input in input.items():
            embeddings[study] = self.dropout(
                self.embedder(self.input_dropout(sub_input)))
        return self.classifier_head(embeddings)