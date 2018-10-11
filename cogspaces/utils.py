from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, \
    accuracy_score


def zip_data(data, target):
    return {study: (data[study], target[study]) for study in data}


def unzip_data(data):
    return {study: data[study][0] for study in data}, \
           {study: data[study][1] for study in data}


def compute_metrics(preds, targets, target_encoder):
    all_f1 = {}
    all_prec = {}
    all_recall = {}
    all_confusion = {}
    all_accuracies = {}
    for study in preds:
        these_preds = preds[study]['contrast']
        these_targets = targets[study]['contrast']
        all_accuracies[study] = accuracy_score(these_preds, these_targets)
        precs, recalls, f1s, support = precision_recall_fscore_support(
            these_preds, these_targets, warn_for=())
        contrasts = target_encoder.le_[study]['contrast'].classes_
        all_prec[study] = {contrast: prec for contrast, prec in
                           zip(contrasts, precs)}
        all_recall[study] = {contrast: recall for contrast, recall in
                             zip(contrasts, recalls)}
        all_f1[study] = {contrast: f1 for contrast, f1 in
                         zip(contrasts, f1s)}
        all_confusion[study] = confusion_matrix(these_preds, these_targets).tolist()
    return {'confusion': all_confusion,
            'prec': all_prec,
            'recall': all_recall,
            'f1': all_f1,
            'accuracies': all_accuracies}


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
