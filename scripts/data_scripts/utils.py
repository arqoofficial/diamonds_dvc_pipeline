import sys
import io
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger


def check_input(
    args_number: int,
    script: str,
    args_list: list[str],
) -> str:
    """Checks sys.argv input and returns correct input path"""
    # check args
    if len(sys.argv) != args_number:
        args = " ".join(args_list)
        logger.error(f"Arguments error. Usage:\t`python3 {script} {args}`")
        logger.warning("SystemExit")
        sys.exit(1)

    # check input file path
    f_input = os.path.abspath(sys.argv[1])
    if not Path(f_input).exists():
        logger.error(
            "diamonds csv not found, please check path correctnes: " f"`{f_input}`"
        )
        raise SystemExit
    return f_input


def handle_exception(e: Exception, message: Optional[str] = None) -> None:
    error_type = type(e).__name__
    logger.error(f"An error occurred: {error_type}. {e}")
    if message:
        logger.warning(message)
    raise SystemExit


def df_from_path(f_input: str, print_info: bool = False) -> pd.DataFrame:
    """Gets csv path and returns df"""
    f_input = os.path.abspath(f_input)
    df = pd.read_csv(f_input)
    logger.success(f"CSV-file was loaded from `{f_input}`")

    UNNAMED_COLUMN = "Unnamed: 0"
    if UNNAMED_COLUMN in df.columns:
        df.drop(columns=UNNAMED_COLUMN, inplace=True)
        logger.debug(f"Column `{UNNAMED_COLUMN}` was dropped")
    if print_info:
        buf = io.StringIO()
        df.info(buf=buf)
        df_info = "\n".join(buf.getvalue().splitlines()[1:])
        logger.info(f"df_info:\n{df_info}")
    return df


def save_csv(df: pd.DataFrame, f_output: str) -> None:
    f_output = os.path.abspath(f_output)
    df.to_csv(f_output, index=False)
    logger.success(f"Result DataFrame was saved as `{f_output}`")


def csv_pipeline(
    func,
    f_input: str,
    f_output: str,
    params: Optional[dict] = None,
) -> None:
    df = df_from_path(f_input)
    if params:
        df = func(df, params)
    else:
        df = func(df)
    save_csv(df, f_output)
