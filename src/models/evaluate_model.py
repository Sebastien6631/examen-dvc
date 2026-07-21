import json
import logging
import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

PROCESSED_DIR = "data/processed_data"
MODELS_DIR = "models"
METRICS_DIR = "metrics"
DATA_DIR = "data"


def main():
    logger = logging.getLogger(__name__)
    logger.info('evaluating trained model on test set')

    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test_scaled.csv")
    y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv").values.ravel()

    with open(f"{MODELS_DIR}/rf_model.pkl", "rb") as f:
        model = pickle.load(f)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    logger.info('mse: %.4f, r2: %.4f', mse, r2)

    predictions_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": predictions,
    })
    predictions_df.to_csv(f"{DATA_DIR}/prediction.csv", index=False)

    scores = {"mse": mse, "r2": r2}
    with open(f"{METRICS_DIR}/scores.json", "w") as f:
        json.dump(scores, f, indent=4)

    logger.info('predictions saved to %s/prediction.csv', DATA_DIR)
    logger.info('scores saved to %s/scores.json', METRICS_DIR)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
