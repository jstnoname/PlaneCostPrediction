from typing import Any

from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

from pandas import DataFrame


class MultimodelLGMB(BaseEstimator, RegressorMixin):
    def __init__(self, models_params: dict[str, dict[str, Any]]):
        self.models_params = models_params
        self.models_ = {}

    def fit(self, X: DataFrame, y: DataFrame) -> 'MultimodelLGMB':
        for tariff, params in self.models_params.items():
            model = LGBMRegressor(**params)
            cur_target_smooth = y[f'target_{tariff}'] + 1
            model.fit(X, cur_target_smooth)
            self.models_[tariff] = model
        return self

    def predict(self, X: DataFrame) -> DataFrame:
        preds = {}
        for tariff, model in self.models_.items():
            pred = model.predict(X) - 1
            preds[f'pred_{tariff}'] = pred.clip(0)
        return DataFrame(preds, index=X.index)


class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, exclude_cols: list[str], cat_cols: list[str]):
        self.exclude_cols = exclude_cols
        self.cat_cols = cat_cols

    def fit(self, X: Any = None, y: Any = None) -> 'DataPreprocessor':
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        X_copy = X.copy()

        cols_to_drop = [col for col in X_copy.columns if col in self.exclude_cols]
        X_copy = X_copy.drop(cols_to_drop, axis=1)

        for col in self.cat_cols:
            if col in X.columns:
                X_copy[col] = X_copy[col].astype('category')

        return X_copy
