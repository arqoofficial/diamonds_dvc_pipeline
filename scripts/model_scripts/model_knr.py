# import os
# import pickle
# import sys

# import pandas as pd
# import yaml
# from sklearn.tree import DecisionTreeClassifier

# if len(sys.argv) != 3:
#     sys.stderr.write("Arguments error. Usage:\n")
#     sys.stderr.write("\tpython dt.py data-file model \n")
#     sys.exit(1)

# f_input = sys.argv[1]
# f_output = os.path.join("models", sys.argv[2])
# os.makedirs(os.path.join("models"), exist_ok=True)

# params = yaml.safe_load(open("params.yaml"))["train"]
# p_seed = params["seed"]
# p_max_depth = params["max_depth"]

# df = pd.read_csv(f_input, header=None)
# X = df.iloc[:, [1, 2, 3]]
# y = df.iloc[:, 0]

# clf = DecisionTreeClassifier(max_depth=p_max_depth, random_state=p_seed)
# clf.fit(X, y)

# with open(f_output, "wb") as fd:
#     pickle.dump(clf, fd)


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

from utils import check_input, csv_pipeline, handle_exception


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
    
    X = df.drop(columns="price")
    y = df[["price"]]
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
    """Main function for pipeline.py script"""
    # paths
    f_input = check_input(2, "pipeline.py", ["data-file"])
    base_path = Path(__file__).parents[2]
    f_output = base_path / "data" / "stage4" / "diamonds.csv"
    f_output.parent.mkdir(parents=True, exist_ok=True)
    f_output = os.path.abspath(f_output)

    # params
    params = dict(yaml.safe_load(open(base_path / "params.yaml"))["preprocessors"])

    # main
    csv_pipeline(preprocessors, f_input, f_output, params)


if __name__ == "__main__":
    main()
    # try:
    #     main()
    # except Exception as e:
    #     handle_exception(e)
