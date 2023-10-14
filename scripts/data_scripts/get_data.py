import os

from kaggle.api.kaggle_api_extended import KaggleApi
from loguru import logger


def handle_exception(e: Exception, message: str) -> None:
    error_type = type(e).__name__
    logger.error(f"An error occurred: {error_type}. {e}")
    logger.warning(message)
    raise SystemExit


if __name__ == "__main__":
    logger.info("Start get_data.py script")

    try:
        api = KaggleApi()
        api.authenticate()
        logger.info("Kaggle authentication was succesfull")
    except Exception as e:
        handle_exception(
            e,
            "You need to provide correct kaggle.json with authentication info",
        )

    try:
        data_path = os.path.abspath("../../data/raw/")
        logger.info("Start Diamonds dataset download")
        api.dataset_download_files(
            dataset="shivam2503/diamonds",
            path=data_path,
            unzip=True,
        )
        logger.success(
            f"Diamonds dataset was successfully downloaded into `{data_path}`"
        )
    except Exception as e:
        handle_exception(
            e,
            "Something went wrong with kaggle dataset download",
        )
