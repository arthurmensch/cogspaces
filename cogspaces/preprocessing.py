"""
Preprocessing helpers for multi-study input.
"""


import warnings
from typing import Dict

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
    """"
    Transformer that numericalize task fMRI data.

    """

    def fit(self, targets: Dict[str, pd.DataFrame]) -> 'MultiTargetEncoder':
        """
        Fit the target encoders necessary for dataframe numericalization.

        Parameters
        ----------
        targets : Dict[str, pd.DataFrame]
            Dictionary of dataframes associated to single studies. Each
            dataframe must contain
            the columns ['study', 'subject', 'contrast', 'study_contrast']

        Returns
        -------
        self: MultiTargetEncoder

        """
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
        """
        Transform named targets into numericalized targets.

        Parameters
        ----------
        targets : Dict[str, pd.DataFrame]
            Dictionary of dataframes associated to single studies. Each
            dataframe must contain
            the columns ['study', 'subject', 'contrast', 'study_contrast']

        Returns
        -------
        numericalized_targets: Dict[str, pd.DataFrame]
            Dictionary of dataframes associated to single studies,
             where each column is numericalized.

        """
        res = {}
        for study, target in targets.items():
            d = self.le_[study]
            res[study] = target.apply(lambda x: d[x.name].transform(x))
        return res

    def inverse_transform(self, targets):
        """
        Transform numericalized targets into named targets.

        Parameters
        ----------
        targets: Dict[str, pd.DataFrame]
            Dictionary of dataframes associated to single studies,
            where each column is numericalized. Each dataframe must contain
            the columns ['study', 'subject', 'contrast', 'study_contrast']

        Returns
        -------
        named_targets : Dict[str, pd.DataFrame]
            Dictionary of dataframes associated to single studies. Each
            dataframe must contain
            the columns ['study', 'subject', 'contrast', 'study_contrast']

        """
        res = {}
        for study, target in targets.items():
            d = self.le_[study]
            res[study] = target.apply(lambda x: d[x.name].inverse_transform(x))
        return res

    @property
    def classes_(self):
        """

        Returns
        -------
        classes_: Dict[List[str]]
            Dictionary of classes list for the contrast `target_encoder`.
        """
        return {study: le['contrast'].classes_ for study, le in
                self.le_.items()}
