import numpy as np

class ScoreCallback:
    def __init__(self, estimator, X, y, score_function):
        self.estimator = estimator
        self.X = X
        self.y = y
        self.score_function = score_function
        self.n_iter_ = []
        self.scores_ = []
        self.ranks_ = []

    def __call__(self, n_iter):
        preds = self.estimator.predict(self.X)
        scores = {}
        coef = self.estimator.coef_
        rank = np.linalg.matrix_rank(coef)
        for study in self.y:
            scores[study] = self.score_function(preds[study], self.y[study])
        self.n_iter_.append(n_iter)
        self.scores_.append(scores)
        self.ranks_.append(rank)
        scores_str = ' '.join('%s: %.3f' % (study, score)
                              for study, score in scores.items())
        scores_str = 'Test score: ' + scores_str
        print(scores_str)
        print('Rank :', rank)



