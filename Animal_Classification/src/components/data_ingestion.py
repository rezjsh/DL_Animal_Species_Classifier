import os
from pathlib import Path
import subprocess
from src.entity.config_entity import DataIngestionConfig
from src.utils.logging_setup import logger
from src.utils.helpers import extract_zip

class DataIngestion:
    """
    A class to manage downloading and extracting a dataset from Kaggle.
    This is an example of a Utility/Service class.
    """
    def __init__(self, config: DataIngestionConfig):
        """
        Initializes the manager with the dataset ID and destination path.
        Args:
            config (DataIngestionConfig): The configuration for the data ingestion.
        """
        self.config = config
        self.kaggle_dir = Path.home() / ".kaggle"

    def _setup_kaggle_credentials(self):
        # Ensure ~/.kaggle exists and kaggle.json is placed there with correct permissions
        if not self.kaggle_dir.exists():
            self.kaggle_dir.mkdir(parents=True, exist_ok=True)

        target = self.kaggle_dir / "kaggle.json"
        if not target.exists():
            if not self.config.kaggle_json_path.exists():
                raise FileNotFoundError(f"Could not find {self.config.kaggle_json_path}")
            self.config.kaggle_json_path.rename(target)
            logger.info(f"Moved {self.config.kaggle_json_path} to {target}")
        os.chmod(target, 0o600)
        logger.info("Kaggle credentials setup complete.")


    def download_and_extract(self):
        """
        Downloads the dataset from Kaggle and extracts it.
        """
        self._setup_kaggle_credentials()
        logger.info(f"Checking for dataset at {self.config.dest_dir}...")
        # Check extraction directory exists and is not empty to skip download
        if self.config.extract_dir.exists() and any(self.config.extract_dir.iterdir()):
            logger.info("Dataset already extracted. Skipping download and extraction.")
            return

        logger.info(f"Dataset not found. Downloading {self.config.source_URL}...")
        try:
            # Use subprocess to run the Kaggle command
            subprocess.run(
                ["kaggle", "datasets", "download", self.config.source_URL,
                "-p", str(self.config.dest_dir),
                "--force"  # Overwrite if exists
                ],
                check=True,
                capture_output=True,
                text=True
            )
            logger.info("Download complete.")

            # Unzip the file
            if self.config.unzip:
                zip_path = self.config.dest_dir / self.config.zip_file_name
                if not zip_path.exists():
                   zip_path = next(self.config.dest_dir.glob("*.zip"), None)
                if zip_path is None:
                    raise FileNotFoundError("Downloaded zip file not found.")
                extract_zip(zip_path, self.config.extract_dir)

        except subprocess.CalledProcessError as e:
            logger.error(f"Error during Kaggle command execution:")
            logger.error(f"  Stdout: {e.stdout}")
            logger.error(f"  Stderr: {e.stderr}")
            raise RuntimeError("Please ensure you have configured your Kaggle API credentials.") from e
        except FileNotFoundError:
            raise RuntimeError("Kaggle command not found. Please ensure it's installed and in your PATH.")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred during dataset handling: {e}")