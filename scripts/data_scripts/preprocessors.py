import os
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
)

from utils import check_input, csv_pipeline, handle_exception, check_file_exists


def get_typed_columns(df: pd.DataFrame) -> tuple:
    """Gets df, returns tuple with names of categorial and numerical columns"""
    cat_columns = []
    num_columns = []

    for column_name in df.columns:
        if (df[column_name].dtypes == object) or (df[column_name].dtypes == "category"):
            cat_columns += [column_name]
        else:
            num_columns += [column_name]

    logger.info(f"categorical columns:\t {cat_columns},\tlen = {len(cat_columns)}")
    logger.info(f"numerical columns:\t  {num_columns},\tlen = {len(num_columns)}")

    return cat_columns, num_columns


def preprocessors(df: pd.DataFrame, params: dict) -> Pipeline:
    scalers = params["scalers"]
    if scalers == "minmax":
        scaler = MinMaxScaler()
    elif scalers == "standard":
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()

    poly_features = PolynomialFeatures(
        degree=params["PF_degree"],
        include_bias=False,
        interaction_only=params["PF_interaction"],
    )

    num_pipe = make_pipeline(poly_features, scaler)

    cat_pipe = make_pipeline(
        OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False)
    )
    column_name = params["target"]
    X = df.drop(columns=column_name)
    y = df[[column_name]]
    cat_columns, num_columns = get_typed_columns(X)

    preprocessors = make_column_transformer(
        (num_pipe, num_columns), (cat_pipe, cat_columns)
    )
    logger.success("Preprocessors are ready")
    
    preprocessors.fit(X)
    logger.info("Preprocessors are fitted")

    X_preprocessors = preprocessors.transform(X)
    logger.success("DataFrame is transormed")
    
    df_preprocessors = pd.DataFrame(
        data=X_preprocessors,
        columns=preprocessors.get_feature_names_out()
    )
    result_df = pd.concat([df_preprocessors, y], axis=1)
    logger.info(f"Result DataFrame shape: {result_df.shape}")
    return result_df


def main():
    """Main function for preprocessors.py script"""
    logger.debug("Script preprocessors.py was started")
    # paths
    f_input = check_file_exists(check_input(2, "preprocessors.py", ["data-file"]))
    base_path = Path(__file__).parents[2]
    f_output = base_path / "data" / "stage4" / "diamonds.csv"
    f_output.parent.mkdir(parents=True, exist_ok=True)
    f_output = os.path.abspath(f_output)

    # params
    params = dict(yaml.safe_load(open(base_path / "params.yaml"))["preprocessors"])

    # main
    csv_pipeline(preprocessors, f_input, f_output, params)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        handle_exception(e)
