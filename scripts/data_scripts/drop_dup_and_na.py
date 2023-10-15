import io
import os
import sys
from pathlib import Path

import pandas as pd
from loguru import logger


def process_data_pandas(f_input, f_output):
    """Gets csv path, processes df and saves processed csv"""

    # read_csv
    df = pd.read_csv(f_input)
    logger.success(f"diamond.csv was loaded from `{f_input}`")

    UNNAMED_COLUMN = "Unnamed: 0"
    if UNNAMED_COLUMN in df.columns:
        df.drop(columns=UNNAMED_COLUMN, inplace=True)
        logger.debug(f"Column `{UNNAMED_COLUMN}` was dropped")
    buf = io.StringIO()
    df.info(buf=buf)
    df_info = "\n".join(buf.getvalue().splitlines()[1:])
    logger.info(f"df_info:\n{df_info}")

    # drop_duplicates
    df_shape_before = df.shape
    logger.info(f"df shape before drop_duplicates(): {df_shape_before}")
    df.drop_duplicates(inplace=True, ignore_index=True)
    df_shape_after = df.shape
    logger.info(f"df shape after drop_duplicates(): {df_shape_after}")
    logger.success(f"Dropped {df_shape_before[0] - df_shape_after[0]} rows")

    # drop_na
    nan_amount = df.isna().sum().sum()
    if nan_amount == 0:
        logger.success("There is no NaN to drop")
    else:
        logger.warning(f"There is/are {nan_amount} NaN(s), they will be dropped")
        df.dropna(inplace=True)
        logger.info(f"Result df shape: {df.shape}")

    # save processed csv
    df.to_csv(f_output, index=False)
    logger.success(
        "Prepared diamonds.csv without NaN and duplicates " f"was saved as `{f_output}`"
    )


def main():
    """Main function for drop_dup_and_na.py script"""
    if len(sys.argv) != 2:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython3 drop_dup_and_na.py data-file\n")
        sys.exit(1)

    f_input = os.path.abspath(sys.argv[1])
    if not Path(f_input).exists():
        logger.error(
            "diamonds csv not found, please check path correctnes: " f"`{f_input}`"
        )
        raise SystemExit
    f_output = Path(__file__).parents[2] / "data" / "stage1" / "diamonds.csv"
    f_output.parent.mkdir(parents=True, exist_ok=True)
    f_output = os.path.abspath(f_output)

    process_data_pandas(f_input, f_output)


if __name__ == "__main__":
    main()
