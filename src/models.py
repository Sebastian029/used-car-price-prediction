"""Preprocessing pipeline, baseline models and a Keras MLP with a scikit-learn interface."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_prep import CATEGORICAL_FEATURES, NUMERIC_FEATURES

RANDOM_STATE = 42


def make_preprocessor():
    """Impute, scale and one-hot encode. Fitted inside the pipeline, so on training data only."""
    numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    categorical = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=10,  # car models seen fewer than 10 times are grouped together
        sparse_output=False,
    )
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


class KerasMLPRegressor(BaseEstimator, RegressorMixin):
    """A small fully connected network that works inside sklearn pipelines and CV.

    The target is standardised internally. Early stopping holds out
    `validation_fraction` of the training data it is given, never the test set.
    """

    def __init__(self, hidden_layers=(128, 64), dropout=0.1, learning_rate=1e-3,
                 batch_size=32, epochs=300, patience=20, validation_fraction=0.1,
                 random_state=RANDOM_STATE, verbose=0):
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.verbose = verbose

    def _build(self, n_features):
        from tensorflow import keras

        model = keras.Sequential([keras.layers.Input(shape=(n_features,))])
        for units in self.hidden_layers:
            model.add(keras.layers.Dense(units, activation="relu"))
            if self.dropout:
                model.add(keras.layers.Dropout(self.dropout))
        model.add(keras.layers.Dense(1))
        model.compile(optimizer=keras.optimizers.Adam(self.learning_rate), loss="mse")
        return model

    def fit(self, X, y):
        from tensorflow import keras

        keras.utils.set_random_seed(self.random_state)
        X = np.asarray(X, dtype="float32")
        y = np.asarray(y, dtype="float32")
        self.y_mean_, self.y_std_ = y.mean(), y.std()

        self.model_ = self._build(X.shape[1])
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=self.patience, restore_best_weights=True)
        self.history_ = self.model_.fit(
            X, (y - self.y_mean_) / self.y_std_,
            validation_split=self.validation_fraction,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stopping],
            verbose=self.verbose,
        ).history
        return self

    def predict(self, X):
        X = np.asarray(X, dtype="float32")
        return self.model_.predict(X, verbose=0).ravel() * self.y_std_ + self.y_mean_


def with_log_target(regressor):
    """Full pipeline: preprocessing + regressor trained on log(price), predicting in INR."""
    return TransformedTargetRegressor(
        regressor=Pipeline([("prep", make_preprocessor()), ("model", regressor)]),
        func=np.log,
        inverse_func=np.exp,
    )


def get_models(mlp_params=None):
    """All models compared in train.py, each a complete pipeline."""
    return {
        "Baseline (median)": with_log_target(DummyRegressor(strategy="median")),
        "Ridge regression": with_log_target(Ridge(alpha=1.0)),
        "Random forest": with_log_target(
            RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE)),
        "Gradient boosting": with_log_target(
            HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
        "MLP (Keras)": with_log_target(KerasMLPRegressor(**(mlp_params or {}))),
    }
