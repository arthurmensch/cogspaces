class ScoreCallback:
    def __init__(self, X, y, score_function):
        self.X = X
        self.y = y
        self.score_function = score_function
        self.n_iter_ = []
        self.scores_ = []
        self.ranks_ = []

    def __call__(self, estimator, n_iter):
        preds = estimator.predict(self.X)
        scores = {}
        study_scores = {}
        all_contrast_scores = {}
        for study in self.y:
            scores[study] = self.score_function(preds[study]['contrast'],
                                                self.y[study]['contrast'])
            study_scores[study] = self.score_function(preds[study]['study'],
                                                      self.y[study]['study'])
            all_contrast_scores[study] = self.score_function(
                preds[study]['all_contrast'],
                self.y[study]['all_contrast'])
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
