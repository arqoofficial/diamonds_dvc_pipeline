import os
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
import json
from loguru import logger
from sklearn.metrics import mean_squared_error as mse, r2_score

from utils import check_input, handle_exception, check_file_exists, df_from_path


def load_model(model_path: str) -> Any:
    """Loads models pickle-file and returns model"""
    with open(model_path, "rb") as fd:
        model = pickle.load(fd)
    logger.success(f"Model {type(model).__name__} was loaded from `{model_path}`")
    return model


def split_df(df: pd.DataFrame, params: dict) -> tuple[pd.DataFrame]:
    """Splits DataFrame to X_df and y_df (target)"""
    target_column = params["target"]
    logger.info(f"Target column name: `{target_column}`")
    X = df.drop(columns=target_column)
    logger.info(f"`X` Dataframe shape: {X.shape}")
    y = df[[target_column]]
    logger.info(f"`y` Dataframe shape: {y.shape}")
    logger.success("Input DataFrame was splitted into `X` and `y`")
    return X, y


def save_score(f_output: str, score: dict) -> None:
    """Saves model score into scores.json file"""
    with open(f_output, "w") as fd:
        json.dump(score, fd)
    logger.success(f"Score(s) was(were) saved into `{f_output}`")


def get_extended_scores(y_test: list, y_prediction: list) -> dict:
    """Counts extended scores and returns dicts with scores"""
    MSE = round(mse(y_test, y_prediction), 2)
    RMSE = round(mse(y_test, y_prediction, squared=False), 2)
    R2 = r2_score(y_test, y_prediction)
    logger.info("Test data extended scores:")
    logger.info(f"MSE: {MSE}")
    logger.info(f"RMSE: {RMSE}")
    logger.info(f"R2: {R2}")
    result = {
        "MSE": MSE,
        "RMSE": RMSE,
        "R2": R2,
    }
    return result


def main():
    """Main function for evaluate.py script"""
    logger.debug("Script evaluate.py was started")
    # paths
    f_input, model_path = check_input(3, "evaluate.py", ["data-file", "model-path"])
    f_input = check_file_exists(f_input)
    model_path = check_file_exists(model_path)
    base_path = Path(__file__).parents[2]
    f_output = base_path / "evaluate" / "score.json"
    f_output_extended = base_path / "evaluate" / "scores_extended.json"
    f_output.parent.mkdir(parents=True, exist_ok=True)
    f_output = os.path.abspath(f_output)

    # params
    params = dict(yaml.safe_load(open(base_path / "params.yaml"))["evaluate"])

    # main
    df = df_from_path(f_input)
    X, y = split_df(df, params)
    model = load_model(model_path)

    y_prediction = model.predict(X)
    extended_scores = get_extended_scores(y, y_prediction)
    save_score(f_output_extended, extended_scores)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        handle_exception(e)
