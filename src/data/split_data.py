import click
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COL = "silica_concentrate"


@click.command()
@click.argument('input_filepath', type=click.Path(exists=False), required=0)
@click.argument('output_filepath', type=click.Path(exists=False), required=0)
def main(input_filepath, output_filepath):
    """ Splits the raw dataset (../raw) into X_train, X_test, y_train, y_test
        and saves them into the processed data folder (../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('splitting raw data into train/test sets')

    input_filepath = input_filepath or "data/raw_data/raw.csv"
    output_filepath = output_filepath or "data/processed_data"

    df = pd.read_csv(input_filepath)
    df = df.drop(columns=["date"])

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info('X_train: %s, X_test: %s', X_train.shape, X_test.shape)

    X_train.to_csv(f"{output_filepath}/X_train.csv", index=False)
    X_test.to_csv(f"{output_filepath}/X_test.csv", index=False)
    y_train.to_csv(f"{output_filepath}/y_train.csv", index=False)
    y_test.to_csv(f"{output_filepath}/y_test.csv", index=False)

    logger.info('train/test datasets saved to %s', output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
