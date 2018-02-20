from collections import defaultdict

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder


class MultiStandardScaler(BaseEstimator, TransformerMixin):
    """Simple wrapper around StandardScaler to handle multipe datasets.

    Attributes
    ----------
    self.sc_: dict, Dictionaries indexed by study, owning all StandardScaler
            for each study

    """
    def fit(self, data):
        self.sc_ = {}
        for study, this_data in data.items():
            self.sc_[study] = StandardScaler().fit(this_data)
        return self

    def transform(self, data):
        transformed = {}
        for study, this_data in data.items():
            transformed[study] = self.sc_[study].transform(this_data)
        return transformed

    def inverse_transform(self, data):
        transformed = {}
        for study, this_data in data.items():
            transformed[study] = self.sc_[study].inverse_transform(this_data)
        return transformed


class MultiTargetEncoder(BaseEstimator, TransformerMixin):
    def fit(self, targets):
        self.le_ = {}
        for study, target in targets.items():
            d = defaultdict(LabelEncoder)
            target.apply(lambda x: d[x.name].fit(x))
            self.le_[study] = d
        return self

    def transform(self, targets):
        res = {}
        for study, target in targets.items():
            d = self.le_[study]
            res[study] = target.apply(lambda x: d[x.name].transform(x))
        return res

    def inverse_transform(self, targets):
        res = {}
        for study, target in targets.items():
            d = self.le_[study]
            res[study] = target.apply(lambda x: d[x.name].inverse_transform(x))
        return res