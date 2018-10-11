import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, \
    accuracy_score


def zip_data(data, target):
    return {study: (data[study], target[study]) for study in data}


def unzip_data(data):
    return {study: data[study][0] for study in data}, \
           {study: data[study][1] for study in data}


def compute_metrics(preds, targets, target_encoder):
    f1_dict = {}
    prec_dict = {}
    recall_dict = {}
    confusion_dict = {}
    accuracy_dict = {}
    bacc_dict = {}
    for study in preds:
        these_preds = preds[study]['contrast']
        these_targets = targets[study]['contrast']
        accuracy_dict[study] = accuracy_score(these_preds, these_targets)
        precs, recalls, f1s, support = precision_recall_fscore_support(
            these_preds, these_targets, warn_for=())
        contrasts = target_encoder.le_[study]['contrast'].classes_
        prec_dict[study] = {contrast: prec for contrast, prec in
                           zip(contrasts, precs)}
        recall_dict[study] = {contrast: recall for contrast, recall in
                             zip(contrasts, recalls)}
        f1_dict[study] = {contrast: f1 for contrast, f1 in
                         zip(contrasts, f1s)}
        confusion = confusion_matrix(these_preds, these_targets)
        confusion_dict[study] = confusion.tolist()

        baccs = baccs_from_confusion(confusion)
        bacc_dict[study] = {contrast: bacc for contrast, bacc in
                           zip(contrasts, baccs)}

    return {'confusion': confusion_dict,
            'prec': prec_dict,
            'recall': recall_dict,
            'f1': f1_dict,
            'bacc': bacc_dict,
            'accuracy': accuracy_dict,
            }


def baccs_from_confusion(C):
    baccs = []
    total = np.sum(C)
    for i in range(len(C)):
        t = np.sum(C[i])
        p = np.sum(C[:, i])
        n = total - p
        tp = C[i, i]
        fp = p - tp
        fn = t - tp
        tn = total - fp - fn - tp
        baccs.append(.5 * (tp / p + tn / n))
    return baccs


class ScoreCallback:
    def __init__(self, Xs, ys, score_function):
        self.Xs = Xs
        self.ys = ys
        self.score_function = score_function
        self.n_iter_ = []
        self.scores_ = []
        self.ranks_ = []

    def __call__(self, estimator, n_iter):
        preds = estimator.predict(self.Xs)
        scores = {}
        study_scores = {}
        study_contrast_scores = {}
        for study in self.ys:
            scores[study] = self.score_function(preds[study]['contrast'],
                                                self.ys[study]['contrast'])
            study_scores[study] = self.score_function(preds[study]['study'],
                                                      self.ys[study]['study'])
            study_contrast_scores[study] = self.score_function(
                preds[study]['study_contrast'],
                self.ys[study]['study_contrast'])
        self.n_iter_.append(n_iter)
        self.scores_.append(scores)
        scores_str = ' '.join('%s: %.3f' % (study, score)
                              for study, score in scores.items())
        scores_str = 'Score: ' + scores_str
        return scores_str


class MultiCallback:
    def __init__(self, callbacks):
        self.callbacks = callbacks

    def __call__(self, *args, **kwargs):
        for name, callback in self.callbacks.items():
            output = callback(*args, **kwargs)
            print('[%s] %s' % (name, output))
