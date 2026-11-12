"""Grid search over MLP hyperparameters with 3-fold CV on the training split only.

Writes results/mlp_grid.csv (all configurations) and results/mlp_best_params.json,
which train.py picks up. The test set is not touched here.
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold

from data_prep import train_test
from models import KerasMLPRegressor, with_log_target

RESULTS = Path(__file__).resolve().parent.parent / "results"

PARAM_GRID = {
    "regressor__model__hidden_layers": [(64, 32), (128, 64), (256, 128, 64)],
    "regressor__model__dropout": [0.0, 0.2],
    "regressor__model__learning_rate": [1e-3, 3e-4],
}


def main():
    RESULTS.mkdir(exist_ok=True)
    X_train, _, y_train, _ = train_test()

    search = GridSearchCV(
        with_log_target(KerasMLPRegressor()),
        PARAM_GRID,
        cv=KFold(n_splits=3, shuffle=True, random_state=42),
        scoring="neg_mean_absolute_error",
        verbose=2,
    )
    search.fit(X_train, y_train)

    grid = pd.DataFrame(search.cv_results_)
    grid = grid[["params", "mean_test_score", "std_test_score", "mean_fit_time"]]
    grid["cv_mae_inr"] = -grid.pop("mean_test_score")
    grid = grid.sort_values("cv_mae_inr")
    grid.to_csv(RESULTS / "mlp_grid.csv", index=False)

    best = {k.split("__")[-1]: v for k, v in search.best_params_.items()}
    (RESULTS / "mlp_best_params.json").write_text(json.dumps(best, indent=2))
    print("Best params:", best, f"CV MAE: {-search.best_score_:,.0f} INR")


if __name__ == "__main__":
    main()
