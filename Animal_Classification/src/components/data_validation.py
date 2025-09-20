import os
from pathlib import Path
from typing import Set
from PIL import Image, UnidentifiedImageError
import json
from datetime import datetime
from src.entity.config_entity import DatasetValidationConfig
from src.utils.logging_setup import logger


class DatasetValidator:
    """
    A class to validate a dataset based on specified criteria.

    This class checks for required classes, minimum samples per class,
    class imbalance ratio, and image integrity (corruption and size).
    """
    def __init__(self, config: DatasetValidationConfig):
        """
        Initializes the DatasetValidator with configuration.

        Args:
            config (DatasetValidationConfig): Configuration object containing
                                             validation parameters.
        """
        self.config = config
        self.dataset_dir = self.config.dataset_dir
        self.report_dir = self.config.report_dir
        # Convert required classes to lowercase for case-insensitive comparison
        self.required_classes = set(cls.lower() for cls in self.config.required_classes)
        self.min_samples_per_class = self.config.min_samples_per_class
        self.max_class_ratio = self.config.max_class_ratio
        # Convert allowed extensions to lowercase for case-insensitive comparison
        self.allowed_extensions = set(ext.lower() for ext in self.config.allowed_extensions)
        self.min_image_size = self.config.min_image_size

        self.errors = []  # List to collect validation error messages
        self.warnings = []  # List for non-critical warnings
        logger.info("DatasetValidator initialized with config.")


    def validate(self) -> bool:
        """
        Performs all validation checks on the dataset.

        Returns:
            bool: True if validation passes without critical errors, False otherwise.
        """
        logger.info("Starting dataset validation.")
        # Check if the dataset directory exists and is valid
        self._check_dataset_dir()
        # If the directory check fails, stop validation and report
        if self.errors:
            logger.error(f"Dataset directory check failed: {self.errors[0]}")
            self.save_report()
            return False

        # Proceed with other checks if the directory is valid
        self._check_classes()
        self._check_samples_per_class()
        self._check_images()

        # Save the final validation report
        self.save_report()

        # Return validation status based on collected errors
        if self.errors:
            logger.error("Dataset validation failed with errors.")
            return False
        else:
            logger.info("Dataset validation successful.")
            return True

    def _check_dataset_dir(self):
        """Checks if the dataset directory exists and is a directory."""
        logger.info(f"Checking dataset directory: {self.dataset_dir}")
        if not self.dataset_dir.exists() or not self.dataset_dir.is_dir():
            self.errors.append(f"Dataset directory '{self.dataset_dir}' does not exist or is not a directory.")
            logger.error(f"Dataset directory check failed for {self.dataset_dir}")
        else:
            logger.info(f"Dataset directory {self.dataset_dir} is valid.")


    def _check_classes(self):
        """Checks if all required classes are present and reports any extra classes."""
        logger.info("Checking for required classes.")
        # Get the names of present directories (assumed to be class names) in lowercase
        present_classes = {d.name.lower() for d in self.dataset_dir.iterdir() if d.is_dir()}
        # Find missing and extra classes compared to the required set
        missing = self.required_classes - present_classes
        extra = present_classes - self.required_classes
        if missing:
            self.errors.append(f"Missing expected classes: {missing}")
            logger.error(f"Missing expected classes: {missing}")
        if extra:
            self.warnings.append(f"Unexpected extra classes found: {extra}")
            logger.warning(f"Unexpected extra classes found: {extra}")
        # Log success if no issues found with classes
        if not missing and not extra:
             logger.info("All required classes are present and no extra classes found.")


    def _check_samples_per_class(self):
        """Checks if each class has minimum samples and evaluates class imbalance."""
        logger.info("Checking sample counts per class.")
        class_counts = {}
        for class_dir in self.dataset_dir.iterdir():
            if class_dir.is_dir():
                # Count files with allowed extensions in each class directory
                count = sum(1 for f in class_dir.iterdir() if f.suffix.lower() in self.allowed_extensions)
                class_counts[class_dir.name] = count
                # Check for minimum samples per class
                if count < self.min_samples_per_class:
                    self.errors.append(f"Class '{class_dir.name}' has only {count} samples, less than minimum {self.min_samples_per_class}")
                    logger.error(f"Class '{class_dir.name}' sample count {count} is less than minimum {self.min_samples_per_class}")

        # Check class imbalance ratio if there are any classes
        if class_counts:
            min_count = min(class_counts.values())
            max_count = max(class_counts.values())
            # Avoid division by zero for ratio calculation
            if min_count == 0:
                ratio = float('inf')
            else:
                ratio = max_count / min_count
            # Check if the ratio exceeds the maximum allowed
            if ratio > self.max_class_ratio:
                self.errors.append(f"Class imbalance too high: ratio {ratio:.2f} (max allowed {self.max_class_ratio})")
                logger.error(f"Class imbalance ratio {ratio:.2f} is too high (max allowed {self.max_class_ratio})")
            else:
                logger.info(f"Class imbalance ratio {ratio:.2f} is within allowed limits.")
        else:
            # Warning if no class directories were found
            logger.warning("No class directories found to check sample counts.")


    def _check_images(self):
        """Checks all images for corruption and minimum size."""
        logger.info("Checking images for corruption and size.")
        corrupt_images = []
        small_images = []
        for class_dir in self.dataset_dir.iterdir():
            if class_dir.is_dir():
                for img_file in class_dir.iterdir():
                    if img_file.suffix.lower() in self.allowed_extensions:
                        try:
                            # Attempt to open the image to check for corruption
                            with Image.open(img_file) as img:
                                # Check image dimensions
                                if img.width < self.min_image_size or img.height < self.min_image_size:
                                    small_images.append(str(img_file))
                        except UnidentifiedImageError:
                            # Catch errors for unreadable or corrupt images
                            corrupt_images.append(str(img_file))
                        except Exception as e:
                            # Log any other unexpected errors during image processing
                            logger.error(f"Error processing image {img_file}: {e}")

        # Report found corrupted images as errors
        if corrupt_images:
            self.errors.append(f"Found {len(corrupt_images)} corrupted or unreadable images.")
            logger.error(f"Found {len(corrupt_images)} corrupted or unreadable images.")
            # Log details of a few corrupted images to avoid overwhelming logs
            for img_path in corrupt_images[:10]: # Log only the first 10 for brevity
                 logger.error(f"Corrupted image: {img_path}")

        # Report found small images as warnings
        if small_images:
            self.warnings.append(f"Found {len(small_images)} images smaller than minimum size {self.min_image_size} pixels.")
            logger.warning(f"Found {len(small_images)} images smaller than minimum size {self.min_image_size} pixels.")
            # Log details of a few small images for brevity
            for img_path in small_images[:10]: # Log only the first 10 for brevity
                 logger.warning(f"Small image: {img_path}")

        # Log success if no image issues found
        if not corrupt_images and not small_images:
            logger.info("No corrupted or small images found.")


    def save_report(self) -> Path:
        """
        Saves the validation errors and warnings to a JSON report file.

        Returns:
            Path: The path to the saved report file.
        """
        logger.info("Saving validation report.")
        # Generate a timestamp for the report filename
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        report_path = self.report_dir / f"{timestamp}-dataset-validation-report.json"

        # Structure the report data
        report = {
            "errors": self.errors,
            "warnings": self.warnings
        }

        try:
            # Save the report data to a JSON file
            with report_path.open("w", encoding="utf-8") as f:
                json.dump(report, f, indent=4)
            logger.info(f"Validation report saved to: {report_path}")
        except Exception as e:
             # Log error if saving the report fails
             logger.error(f"Error saving validation report to {report_path}: {e}")
             # Optionally add error to self.errors if report saving is critical
             # self.errors.append(f"Failed to save validation report: {e}")

        return report_path