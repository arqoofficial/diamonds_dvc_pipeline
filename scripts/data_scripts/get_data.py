from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from loguru import logger


def handle_exception(e: Exception, message: str) -> None:
    error_type = type(e).__name__
    logger.error(f"An error occurred: {error_type}. {e}")
    logger.warning(message)
    raise SystemExit


def auth_kaggle() -> KaggleApi:
    """Authentication in KaggleApi"""
    try:
        api = KaggleApi()
        api.authenticate()
        logger.info("Kaggle authentication was succesfull")
    except Exception as e:
        handle_exception(
            e,
            "You need to provide correct kaggle.json with authentication info",
        )
    return api


def download_dataset(api: KaggleApi) -> None:
    """Download Diamonds.csv dataset into ../../data/raw/ folder"""
    data_path = Path(__file__).parents[2] / "data" / "raw"
    logger.info("Start Diamonds dataset download")
    try:
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


if __name__ == "__main__":
    logger.info("Start get_data.py script")
    download_dataset(auth_kaggle())
