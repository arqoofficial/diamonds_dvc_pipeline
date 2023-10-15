import os
from pathlib import Path

import pandas as pd
from loguru import logger

from utils import handle_exception, csv_pipeline, check_input


def drop_na(df: pd.DataFrame) -> pd.DataFrame:
    """Gets df and retuns df without NaN row(s)"""
    nan_amount = df.isna().sum().sum()
    if nan_amount == 0:
        logger.success("There is no NaN to drop")
    else:
        logger.warning(f"There is/are {nan_amount} NaN(s), they will be dropped")
        df.dropna(inplace=True)
        logger.info(f"Result df shape: {df.shape}")
    return df


def main():
    """Main function for drop_na.py script"""
    # paths
    f_input = check_input(2, "drop_na.py", ["data-file"])
    f_output = Path(__file__).parents[2] / "data" / "stage2" / "diamonds.csv"
    f_output.parent.mkdir(parents=True, exist_ok=True)
    f_output = os.path.abspath(f_output)

    # main
    csv_pipeline(drop_na, f_input, f_output)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        handle_exception(e)
