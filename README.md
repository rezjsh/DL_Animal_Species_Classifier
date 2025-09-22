# DL_Animal_Species_Classifier

Here is a complete and customized README for your DL_Animal_Species_Classifier project, structured for GitHub:

---

# DL Animal Species Classifier

Advanced Deep Learning for recognizing animal species from images, featuring a modular ML pipeline, experiment configs, a robust Flask web app, and Docker support.

---

## Table of Contents

- [DL\_Animal\_Species\_Classifier](#dl_animal_species_classifier)
- [DL Animal Species Classifier](#dl-animal-species-classifier)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Features](#features)
  - [Tech Stack](#tech-stack)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Training and Evaluation](#training-and-evaluation)
  - [Inference: Web App](#inference-web-app)
  - [Inference: Script/Batch](#inference-scriptbatch)
  - [Docker](#docker)
  - [Logging](#logging)
  - [Results and Reports](#results-and-reports)
  - [Troubleshooting](#troubleshooting)
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)

---

## Overview

DL Animal Species Classifier brings together a complete deep learning workflow to classify animal images, from ingestion to prediction. It leverages modern TensorFlow/Keras models, well-structured configuration, reproducible data processing, and interactive web-based inference via Flask. This project is ideal for data scientists, researchers, and developers aiming to experiment with or deploy animal image classifiers in production or educational environments.

---

## Features

- Fully modular pipeline (data ingest, validation, training, evaluation, prediction)
- YAML-based experiment and pipeline configuration
- Model architectures: EfficientNet, ResNet, and custom CNNs
- Automated logging and error tracking
- Flask web app for simple, browser-based predictions
- Centralized result and artifact management

---

## Tech Stack

- Python 3.8+
- TensorFlow/Keras
- Flask
- NumPy, Pandas, Matplotlib, Seaborn
- PyYAML, Box

---

## Project Structure

```
DL_Animal_Species_Classifier/
├─  Animal_Classification/
│   ├─ config/
│   │  ├─ config.yaml
│   ├─ data/
│   │  ├─ 01_raw/
│   ├─ docs/
│   ├─ logs/
│   ├─ models/
│   │  └─ evaluation/
│   ├─ notebooks/
│   ├─ reports/
│   │  └─ figures/
│   ├─ tests/
│   ├─ src/
│   │  ├─ components/
│   │  │  ├─ __init__.py
│   │  │  ├─ data_ingestion.py
│   │  │  ├─ data_validation.py
│   │  │  ├─ model_evaluation.py
│   │  │  ├─ model_trainer.py
│   │  │  ├─ model.py
│   │  │  ├─ prediction.py
│   │  │  └─ prepare_dataset.py
│   │  ├─ config/
│   │  │  ├─ __init__.py
│   │  │  └─ configuration.py
│   │  ├─ constants/
│   │  │  ├─ __init__.py
│   │  │  └─ constants.py
│   │  ├─ core/
│   │  ├─ entity/
│   │  │  ├─ __init__.py
│   │  │  └─ config_entity.py
│   │  ├─ models/
│   │  │  ├─ __init__.py
│   │  │  ├─ base_model.py
│   │  │  ├─ CustomCNNModel.py
│   │  │  ├─ EfficientNetModel.py
│   │  │  ├─ ModelFactory.py
│   │  │  └─ REsnetModel.py
│   │  ├─ modules/
│   │  │  ├─ __init__.py
│   │  │  ├─ augmentation.py
│   │  │  └─ callbacks.py
│   │  ├─ pipeline/
│   │  │  ├─ __init__.py
│   │  │  ├─ stage_01_data_ingestion.py
│   │  │  ├─ stage_02_data_validation.py
│   │  │  ├─ stage_03_prepare_dataset.py
│   │  │  ├─ stage_04_model.py
│   │  │  ├─ stage_05_model_trainer.py
│   │  │  ├─ stage_06_model_evaluation.py
│   │  │  └─ stage_07_model_prediction.py
│   │  └─ utils/
│   │     ├─ __init__.py
│   │     ├─ logging_setup.py
│   │     └─ helpers.py
│   ├─ .env.example
│   ├─ Dockerfile
│   ├─ environment.yml
│   ├─ main.py
│   └─ params.yaml
├─ .gitignore
├─ LICENCE
├─ README.md
├─ requirements.txt
├─ run.py
├─ setup.py
└─ template.py 
```

---

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/DL_Animal_Species_Classifier.git
cd DL_Animal_Species_Classifier
```

2. **Create and activate environment**

Conda (recommended):
```bash
conda env create -f environment.yml
conda activate animal-classifier
```

pip/venv:
```bash
python -m venv .venv
# On Windows
.\\.venv\\Scripts\\activate
# On Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
```

3. **Configure environment variables**

- Copy `.env.example` to `.env` if needed, and fill in required secrets or configuration.

4. **Kaggle Credentials (for dataset download)**

- Place your `kaggle.json` as specified in `Animal_Classification/config/config.yaml` (`data_ingestion.kaggle_json_path`).

---

## Configuration

Edit pipeline behaviors, data paths, and model settings in:
```
Animal_Classification/config/config.yaml
```
**Example options:**
```yaml
model_prediction:
  model_path: models/best_model.keras
  class_names: ["cat", "dog", "wildlife"]
  confidence_threshold: 0.5
```
- Set model save/load paths, classes, thresholds.
- Specify data sources and local directories.

---

## Training and Evaluation

Run the complete ML pipeline (from data ingestion to evaluation and test predictions) using:
```bash
python -m Animal_Classification.main
```
Artifacts will appear in:
- `models/` — Model snapshots
- `reports/` — Plots, metrics, and validation results
- `logs/` — Training and inference logs

---

## Inference: Web App

To start the Flask application for browser-based image predictions:
```bash
python run.py
```
- Visit: http://localhost:5000  
- Upload a JPG/PNG/GIF animal image for instant classification results

The web app uses the model specified in your configuration or default (`models/best_model.keras`).

---

## Inference: Script/Batch

You may run prediction as a script or use the provided prediction pipeline:
```python
from Animal_Classification.app.predictor import AnimalPredictor

predictor = AnimalPredictor('models/best_model.keras')
result = predictor.predict('path/to/your/image.jpg')
print(result)
```

---

## Docker

Build the image:
```bash
docker build -t animal-classifier:latest .
```

Run the container:
```bash
docker run --rm -p 5000:5000 animal-classifier:latest
```
- Access the web app on http://localhost:5000
- Mount your models/data into the container with Docker `-v` options if you want persistence.
## Logging

Logs are written to:
- `logs/running_logs.log` (file)
- Console (stdout)

Configure details in:
- `Animal_Classification/src/utils/logging_setup.py`
- `Animal_Classification/config/logging_config.yaml` (if used)

---

## Results and Reports

- Training curves, metrics, and confusion matrices appear in `reports/*`
- Model checkpoints: `models/`
- Example sample predictions: `reports/evaluation/`

---

## Troubleshooting

- **Model not found in web app:** Set `model_path` in app/config, and ensure model file exists before app start.
- **Kaggle errors:** Double-check `kaggle.json` placement and Kaggle API status.
- **GPU/TensorFlow issues:** Review logs, use supported CUDA/cuDNN version, and enable memory growth where appropriate.
- **Path errors:** Windows users: try raw string or double backslash paths.
- **Permission denied:** Use correct file/directory permissions and ownership for all project folders/files, especially within Docker.

---

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork this repository
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Create a new Pull Request

---

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for more details.

---

## Acknowledgements

- Dataset: Animals-10 from [Kaggle](https://www.kaggle.com/alessiocorrado99/animals10)
- Keras, TensorFlow, Flask contributors
- Open-source ML/DL community

---

This README aims to give a clean, professional onboarding experience for contributors and users, with real project paths and accurate instructions.
