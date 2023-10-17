import os
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from loguru import logger
from sklearn.neighbors import KNeighborsRegressor

from utils import check_input, handle_exception


def dump_model(model: Any, f_output: str) -> None:
    """Dumps models into pickle-file"""
    with open(f_output, "wb") as fd:
        pickle.dump(model, fd)
    logger.success(f"Model {type(model).__name__} was saved as `{f_output}`")


def prepare_knr(df: pd.DataFrame, params: dict) -> Any:
    """Gets DataFrame and params and prepares KNeighborsRegressor with specified params"""
    model = KNeighborsRegressor(
        n_neighbors=params["n_neighbors"], 
        weights=params["weights"],
        n_jobs=-1
    )
    target_column = params["target"]
    X = df.drop(columns=target_column)
    y = df[[target_column]]
    model.fit(X, y)
    logger.success(f"Model {type(model).__name__} was fitted")
    return model


def main():
    """Main function for model_knr.py script"""
    logger.debug("Script model_knr.py was started")
    # paths
    f_input, model_name = check_input(3, "model_knr.py", ["data-file", "model"])
    base_path = Path(__file__).parents[2]
    f_output = base_path / "models" / f"{model_name}.pkl"
    f_output.parent.mkdir(parents=True, exist_ok=True)
    f_output = os.path.abspath(f_output)

    # params
    params = dict(yaml.safe_load(open(base_path / "params.yaml"))["train"])

    # main
    df = pd.read_csv(f_input)
    logger.info(f"CSV-file was read from `{f_input}`")
    model = prepare_knr(df, params)
    dump_model(model, f_output)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        handle_exception(e)
