from pathlib import Path
from src.constants.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.entity.config_entity import DataValidationConfig
from src.utils.helpers import create_directory, read_yaml_file
from src.utils.logging_setup import logger
from src.core.singlton import SingletonMeta

class ConfigurationManager(metaclass=SingletonMeta):
    def __init__(self, config_file_path: str = CONFIG_FILE_PATH, params_file_path: str = PARAMS_FILE_PATH):
        self.config = read_yaml_file(config_file_path)
        self.params = read_yaml_file(params_file_path)


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        logger.info("Getting data ingestion config")
        config = self.config.data_ingestion
        dirs_to_create = [config.dest_dir, config.extract_dir]
        create_directory(dirs_to_create)
        logger.info(f"Directories created: {dirs_to_create}")
        logger.info(f"Data ingestion config: {config}")
        data_ingestion_config = DataIngestionConfig(
            kaggle_json_path=Path(config.kaggle_json_path),
            dest_dir=Path(config.dest_dir),
            extract_dir=Path(config.extract_dir),
            source_URL=config.source_URL,
            unzip=config.unzip,
            zip_file_name=config.zip_file_name)

        logger.info(f"Data ingestion config created: {data_ingestion_config}")
        return data_ingestion_config

    def get_data_validation_config(self) -> DatasetValidationConfig:
        logger.info("Getting data validation config")
        config = self.config.dataset_validation
        params = self.params.dataset_validation
        dirs_to_create = [config.report_dir]
        create_directory(dirs_to_create)
        logger.info(f"Directories created: {dirs_to_create}")
        logger.info(f"Data validation config: {config}")
        data_validation_config = DatasetValidationConfig(
            dataset_dir=Path(config.dataset_dir),
            report_dir=Path(config.report_dir),
            required_classes=params.required_classes,
            min_samples_per_class=params.min_samples_per_class,
            min_image_size=params.min_image_size,
            max_class_ratio=params.max_class_ratio,
            allowed_extensions=params.allowed_extensions
        )
        logger.info(f"Data validation config created: {data_validation_config}")
        return data_validation_config

    def get_dataset_preparation_config(self) -> DataPreparationConfig:
        logger.info("Getting dataset preparation config")
        config = self.config.dataset
        params = self.params.dataset
        logger.info(f"Dataset preparation config: {config}")
        dataset_preparation_config = DataPreparationConfig(
            data_dir=Path(config.data_dir),
            img_size=params.img_size,
            batch_size=params.batch_size,
            shuffle_buffer_size=params.shuffle_buffer_size,
            validation_split=params.validation_split
        )
        logger.info(f"Dataset preparation config created: {dataset_preparation_config}")
        return dataset_preparation_config


    def get_augmentation_config(self) -> AugmentationConfig:
          logger.info("Getting augmentation config")
          params = self.params.augmentation
          config = self.config.augmentation
          create_directory([Path(config.save_path).parent])
          logger.info(f"Augmentation directories created: {[Path(config.save_path).parent]}")
          logger.info(f"Augmentation config: {params}")
          augmentation_config = AugmentationConfig(
              horizontal_flip=params.horizontal_flip,
              vertical_flip=params.vertical_flip,
              rotation_range=params.rotation_range,
              width_shift_range=params.width_shift_range,
              height_shift_range=params.height_shift_range,
              zoom_range=params.zoom_range,
              contrast_range=params.contrast_range,
              brightness_factor=params.brightness_factor,
              save_path=Path(config.save_path),
              num_images=params.num_images,
          )
          logger.info(f"Augmentation config created: {augmentation_config}")
          return augmentation_config


    def get_animal_species_classifier_model_config(self) -> AnimalSpeciesClassifierModelConfig:
            logger.info("Getting animal species classifier model config")
            params = self.params.model
            logger.info(f"Animal species classifier model params: {params}")
            animal_species_classifier_model_config = AnimalSpeciesClassifierModelConfig(
                model_choice=params.model_choice,
                img_size=params.img_size,
                batch_size=params.batch_size,
                shuffle_buffer_size=params.shuffle_buffer_size,
                validation_split=params.validation_split,
                learning_rate=params.learning_rate,
                # num_classes=params.num_classes,
                num_layers_to_unfreeze=params.num_layers_to_unfreeze,
                augmentation_enabled=params.augmentation_enabled
            )
            logger.info(f"Animal species classifier model config created: {animal_species_classifier_model_config}")
            return animal_species_classifier_model_config


    def get_custom_callbacks_config(self) -> CustomCallbacksConfig:
            logger.info("Getting custom callbacks config")
            config = self.config.callbacks
            params = self.params.callbacks
            create_directory([config.checkpoint_dir])
            logger.info(f"Custom callbacks config: {config}")
            custom_callbacks_config = CustomCallbacksConfig(
                patience=params.patience,
                checkpoint_dir=config.checkpoint_dir,
                min_delta=params.min_delta,
                # Safely get lr_schedule with a default of None
                lr_schedule=params.get('lr_schedule', None),
                verbose=params.verbose
            )
            logger.info(f"Custom callbacks config created: {custom_callbacks_config}")
            return custom_callbacks_config


    def get_model_training_config(self) -> ModelTrainingConfig:
            logger.info("Getting model training config")
            config = self.config.model_training
            params = self.params.model_training
            dirs_to_create = [config.save_dir, config.save_plot_dir]
            create_directory(dirs_to_create)
            logger.info(f"Model training directories created: {dirs_to_create}")
            logger.info(f"Model training config: {config}")
            model_training_config = ModelTrainingConfig(
                model_name=params.model_name,
                epochs=params.epochs,
                batch_size=params.batch_size,
                learning_rate=params.learning_rate,
                callbacks=params.callbacks,
                save_dir=config.save_dir,
                save_plot_dir=config.save_plot_dir,
                verbose = params.verbose
            )
            logger.info(f"Model training config created: {model_training_config}")
            return model_training_config


    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        logger.info("Getting model evaluation config")
        config = self.config.evaluation
        create_directory([config.save_dir])
        logger.info(f"Model evaluation directories created: {config.save_dir}")
        logger.info(f"Model evaluation config: {config}")
        model_evaluation_config = ModelEvaluationConfig(
            save_dir=config.save_dir,
            model_path=config.model_path,
            saved_model_name=config.saved_model_name
        )
        logger.info(f"Model evaluation config created: {model_evaluation_config}")
        return model_evaluation_config



    def get_model_prediction_config(self) -> ModelPredictionConfig:
        logger.info("Getting model prediction config")
        config = self.config.model_prediction
        logger.info(f"Model prediction config: {config}")
        model_prediction_config = ModelPredictionConfig(
            model_path=config.model_path,
            class_names=config.class_names,
            confidence_threshold=config.confidence_threshold
        )
        logger.info(f"Model prediction config created: {model_prediction_config}")
        return model_prediction_config