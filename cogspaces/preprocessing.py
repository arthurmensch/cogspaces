import warnings

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings('ignore', category=DeprecationWarning,
                        module=r'sklearn.preprocessing.label.*')


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
            # self.sc_[study].scale_ /= np.sqrt(len(this_data))
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

    @property
    def scale_(self):
        return {study: sc.scale_ for study, sc in self.sc_.items()}

    @property
    def mean_(self):
        return {study: sc.mean_ for study, sc in self.sc_.items()}


class MultiTargetEncoder(BaseEstimator, TransformerMixin):
    def fit(self, targets):
        self.le_ = {}

        study_contrasts = pd.concat([target['study_contrast']
                                   for target in targets.values()])
        studies = pd.concat([target['study'] for target in targets.values()])
        le_study_contrast = LabelEncoder().fit(study_contrasts)
        le_study = LabelEncoder().fit(studies)
        for study, target in targets.items():
            self.le_[study] = dict(
                contrast=LabelEncoder().fit(target['contrast']),
                subject=LabelEncoder().fit(target['subject']),
                study_contrast=le_study_contrast,
                study=le_study,
                )
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

    @property
    def classes_(self):
        return {study: le['contrast'].classes_ for study, le in
                self.le_.items()}
