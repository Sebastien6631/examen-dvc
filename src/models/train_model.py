import logging
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

PROCESSED_DIR = "data/processed_data"
MODELS_DIR = "models"


def main():
    logger = logging.getLogger(__name__)
    logger.info('training RandomForestRegressor with best params')

    X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train_scaled.csv")
    y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train.csv").values.ravel()

    with open(f"{MODELS_DIR}/best_params.pkl", "rb") as f:
        best_params = pickle.load(f)
    logger.info('loaded best params: %s', best_params)

    model = RandomForestRegressor(random_state=42, **best_params)
    model.fit(X_train, y_train)

    with open(f"{MODELS_DIR}/rf_model.pkl", "wb") as f:
        pickle.dump(model, f)

    logger.info('trained model saved to %s/rf_model.pkl', MODELS_DIR)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
