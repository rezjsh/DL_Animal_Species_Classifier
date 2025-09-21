from dataclasses import dataclass
from pathlib import Path
from typing import Set, List

@dataclass
class DataIngestionConfig:
    kaggle_json_path: Path
    dest_dir: Path
    extract_dir: Path
    source_URL: str
    unzip: bool
    zip_file_name: str

@dataclass
class DatasetValidationConfig:
    dataset_dir: Path
    report_dir: Path
    required_classes: Set[str]
    min_samples_per_class: int
    max_class_ratio: float
    allowed_extensions: Set[str]
    min_image_size: int

@dataclass
class DataPreparationConfig:
    data_dir: Path
    img_size: int
    batch_size: int
    shuffle_buffer_size: int
    validation_split: float

@dataclass
class AugmentationConfig:
    horizontal_flip: bool
    vertical_flip: bool
    rotation_range: int
    width_shift_range: float
    height_shift_range: float
    zoom_range: float
    contrast_range: float
    brightness_factor: float
    save_path: Path
    num_images: int

@dataclass
class AnimalSpeciesClassifierModelConfig:
    model_choice: str
    img_size: int
    batch_size: int
    shuffle_buffer_size: int
    validation_split: float
    learning_rate: float
    # num_classes: int
    augmentation_enabled: bool
    num_layers_to_unfreeze: int

@dataclass
class CustomCallbacksConfig:
    patience: int
    checkpoint_dir: str
    min_delta: float
    lr_schedule: callable
    verbose: bool


@dataclass
class ModelTrainingConfig:
    model_name: str
    epochs: int
    batch_size: int
    learning_rate: float
    callbacks: list
    save_dir: str
    save_plot_dir: str
    verbose: int


@dataclass
class ModelEvaluationConfig:
    save_dir: Path
    model_path: Path
    saved_model_name: str


@dataclass
class ModelPredictionConfig:
    model_path: Path
    class_names: list[str]
    confidence_threshold: float
