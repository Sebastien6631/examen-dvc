import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROCESSED_DIR = "data/processed_data"


def main():
    logger = logging.getLogger(__name__)
    logger.info('normalizing train/test feature sets')

    X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv")
    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns
    )
    logger.info('fitted StandardScaler on %s rows', len(X_train))

    X_train_scaled.to_csv(f"{PROCESSED_DIR}/X_train_scaled.csv", index=False)
    X_test_scaled.to_csv(f"{PROCESSED_DIR}/X_test_scaled.csv", index=False)

    logger.info('scaled datasets saved to %s', PROCESSED_DIR)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
