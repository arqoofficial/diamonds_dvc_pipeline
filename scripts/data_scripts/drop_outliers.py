import os
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

from utils import check_input, csv_pipeline, handle_exception


def drop_outliers(df: pd.DataFrame, params: dict) -> None:
    """Gets df and retuns df without outliers according to params-dict"""
    total_dropped = 0
    for feature, borders in params.items():
        _min = borders["min"]
        _max = borders["max"]
        logger.debug(f"For feature `{feature}` drop outliers not in [{_min}, {_max}]")

        df_shape_before = df.shape
        logger.info(
            f"df shape before drop_outliers() from `{feature}`: {df_shape_before}"
        )

        question_index = df[(df[feature] < _min) | (df[feature] > _max)].index
        df = df.drop(question_index)
        df_shape_after = df.shape
        logger.info(
            f"df shape after drop_outliers() from `{feature}`: {df_shape_after}"
        )

        dropped = df_shape_before[0] - df_shape_after[0]
        total_dropped += dropped
        logger.success(f"Dropped {dropped} rows(s) for feature `{feature}`")
    logger.info(f"drop_outliers() dropped {total_dropped} row(s) in total")
    return df


def main():
    """Main function for drop_outliers.py script"""
    logger.debug("Script drop_outliers.py was started")
    # paths
    f_input = check_input(2, "drop_outliers.py", ["data-file"])
    base_path = Path(__file__).parents[2]
    f_output = base_path / "data" / "stage3" / "diamonds.csv"
    f_output.parent.mkdir(parents=True, exist_ok=True)
    f_output = os.path.abspath(f_output)

    # params
    params = dict(yaml.safe_load(open(base_path / "params.yaml"))["drop_outliers"])

    # main
    csv_pipeline(drop_outliers, f_input, f_output, params)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        handle_exception(e)
