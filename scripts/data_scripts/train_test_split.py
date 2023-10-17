from pathlib import Path

import pandas as pd
import yaml
from loguru import logger
from sklearn.model_selection import train_test_split

from utils import check_input, df_from_path, handle_exception


def split_df(
    df: pd.DataFrame,
    target_column: str,
    params: dict,
    train_output: str,
    test_output: str,
) -> None:
    """Splits input df"""
    logger.info(f"Start split_df() function with target column `{target_column}`")
    X = df.drop(columns=[target_column])
    y = df[[target_column]]

    split_ratio = params["split_ratio"]
    random_seed = params["seed"]
    logger.info(f"Split ratio = {split_ratio}")
    logger.info(f"Random seed = {random_seed}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split_ratio,
        random_state=random_seed,
    )
    logger.info(f"Length of train df = {len(X_train)}")
    logger.info(f"Length of test df = {len(X_test)}")

    for X_df, y_df, path, text in [
        (X_train, y_train, train_output, "train"),
        (X_test, y_test, test_output, "test"),
    ]:
        pd.concat([X_df, y_df], axis=1).to_csv(path, index=False)
        logger.success(f"{text}.csv was successfully saved as `{path}`")


def main():
    """Main function for train_test_split.py script"""
    logger.debug("Script train_test_split.py was started")
    # paths
    f_input = check_input(2, "train_test_split.py", ["data-file"])
    base_path = Path(__file__).parents[2]
    stage_path = base_path / "data" / "stage5"
    stage_path.mkdir(parents=True, exist_ok=True)

    f_output_train = stage_path / "train.csv"
    f_output_test = stage_path / "test.csv"

    # params
    params = dict(yaml.safe_load(open(base_path / "params.yaml"))["split"])

    # main
    df = df_from_path(f_input)
    split_df(
        df=df,
        target_column="price",
        params=params,
        train_output=f_output_train,
        test_output=f_output_test,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        handle_exception(e)
