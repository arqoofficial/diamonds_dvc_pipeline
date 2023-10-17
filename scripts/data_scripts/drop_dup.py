import os
from pathlib import Path

import pandas as pd
from loguru import logger

from utils import check_input, csv_pipeline, handle_exception


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Gets df and retuns df without duplicates"""
    df_shape_before = df.shape
    logger.info(f"df shape before drop_duplicates(): {df_shape_before}")
    df.drop_duplicates(inplace=True, ignore_index=True)
    df_shape_after = df.shape
    logger.info(f"df shape after drop_duplicates(): {df_shape_after}")
    logger.success(f"Dropped {df_shape_before[0] - df_shape_after[0]} rows")
    return df


def main():
    """Main function for drop_dup.py script"""
    logger.debug("Script drop_dup.py was started")
    # paths
    f_input = check_input(2, "drop_dup.py", ["data-file"])
    f_output = Path(__file__).parents[2] / "data" / "stage1" / "diamonds.csv"
    f_output.parent.mkdir(parents=True, exist_ok=True)
    f_output = os.path.abspath(f_output)

    # main
    csv_pipeline(drop_duplicates, f_input, f_output)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        handle_exception(e)
