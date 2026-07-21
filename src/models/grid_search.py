import logging
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

PROCESSED_DIR = "data/processed_data"
MODELS_DIR = "models"

PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5, 10],
}


def main():
    logger = logging.getLogger(__name__)
    logger.info('running grid search for RandomForestRegressor')

    X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train_scaled.csv")
    y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train.csv").values.ravel()

    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=PARAM_GRID,
        cv=5,
        scoring="r2",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    logger.info('best params: %s', grid_search.best_params_)
    logger.info('best cv r2 score: %.4f', grid_search.best_score_)

    with open(f"{MODELS_DIR}/best_params.pkl", "wb") as f:
        pickle.dump(grid_search.best_params_, f)

    logger.info('best params saved to %s', MODELS_DIR)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
