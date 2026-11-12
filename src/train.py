"""Train every model, score it with 5-fold CV on the training split and once on the test split.

Outputs in results/: metrics.csv, metrics.md, predicted_vs_actual.png,
mlp_learning_curve.png, feature_importance.png.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import KFold, cross_val_score

from data_prep import train_test
from models import RANDOM_STATE, get_models

RESULTS = Path(__file__).resolve().parent.parent / "results"


def load_mlp_params():
    path = RESULTS / "mlp_best_params.json"
    return json.loads(path.read_text()) if path.exists() else None


def test_metrics(y_true, y_pred):
    return {
        "Test MAE (INR)": mean_absolute_error(y_true, y_pred),
        "Test RMSE (INR)": np.sqrt(mean_squared_error(y_true, y_pred)),
        "Test MAPE (%)": 100 * mean_absolute_percentage_error(y_true, y_pred),
        "Test R2": r2_score(y_true, y_pred),
    }


def to_markdown(metrics):
    table = metrics.copy()
    for col in table.columns:
        if "INR" in col:
            table[col] = table[col].map("{:,.0f}".format)
        elif "%" in col:
            table[col] = table[col].map("{:.1f}".format)
        elif col != "Model":
            table[col] = table[col].map("{:.3f}".format)
    return table.to_markdown(index=False)


def plot_predictions(y_test, predictions, path):
    fig, axes = plt.subplots(1, len(predictions), figsize=(6 * len(predictions), 5))
    for ax, (name, y_pred) in zip(np.atleast_1d(axes), predictions.items()):
        ax.scatter(y_test, y_pred, s=8, alpha=0.4)
        lims = [y_test.min(), y_test.max()]
        ax.plot(lims, lims, "r--", lw=1)
        ax.set(xscale="log", yscale="log", title=name,
               xlabel="Actual price (INR)", ylabel="Predicted price (INR)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_learning_curve(history, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["loss"], label="train")
    ax.plot(history["val_loss"], label="validation")
    ax.set(title="MLP learning curve (standardised log price)", xlabel="Epoch", ylabel="MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_importance(model, X_test, y_test, path):
    result = permutation_importance(model, X_test, y_test, scoring="r2",
                                    n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
    order = result.importances_mean.argsort()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(X_test.columns[order], result.importances_mean[order],
            xerr=result.importances_std[order])
    ax.set(title="Permutation importance (drop in test R2)", xlabel="Mean decrease in R2")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    RESULTS.mkdir(exist_ok=True)
    X_train, X_test, y_train, y_test = train_test()
    print(f"Train: {len(X_train)} rows, test: {len(X_test)} rows")

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows, fitted, predictions = [], {}, {}
    for name, model in get_models(load_mlp_params()).items():
        print(f"-> {name}")
        cv_r2 = cross_val_score(clone(model), X_train, y_train, cv=cv, scoring="r2")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        fitted[name], predictions[name] = model, y_pred
        rows.append({"Model": name, "CV R2 (mean)": cv_r2.mean(), "CV R2 (std)": cv_r2.std(),
                     **test_metrics(y_test, y_pred)})

    metrics = pd.DataFrame(rows).sort_values("Test MAE (INR)")
    metrics.to_csv(RESULTS / "metrics.csv", index=False)
    (RESULTS / "metrics.md").write_text(to_markdown(metrics))
    print(metrics.to_string(index=False))

    best = next(n for n in metrics["Model"] if n != "MLP (Keras)")
    plot_predictions(y_test, {best: predictions[best], "MLP (Keras)": predictions["MLP (Keras)"]},
                     RESULTS / "predicted_vs_actual.png")
    plot_learning_curve(fitted["MLP (Keras)"].regressor_.named_steps["model"].history_,
                        RESULTS / "mlp_learning_curve.png")
    plot_importance(fitted[best], X_test, y_test, RESULTS / "feature_importance.png")


if __name__ == "__main__":
    main()
