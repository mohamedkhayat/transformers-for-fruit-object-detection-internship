# Fruit Internship Project

Transformer-based object detection pipeline for fruit localization, built as part of a university internship. The current version supports object detection only and is designed to be extended into a multi-task learning setup (e.g., for weight and volume prediction).

---

## Overview

This repository provides a modular and reproducible deep learning pipeline for object detection using transformer models from Hugging Face. It is built with scalability and maintainability in mind, following best practices in code structure, configuration, and logging.

Key components:
- Object detection using RT-DETR v2
- Configurable training via Hydra
- Advanced augmentations with Albumentations
- Experiment tracking with Weights & Biases
- Early stopping and mixed precision training
- Class-imbalanced sampling strategies
- Sphinx-based auto-generated documentation

---

## Directory Structure

```
mohamedkhayat-fruit_internship/
├── conf/                   # Hydra configuration files
│   ├── config.yaml         # Base training config
│   └── model/              # Model-specific configs
├── docs/                   # Sphinx documentation
│   ├── conf.py             # Sphinx configuration
│   └── index.rst           # Main documentation file
├── src/fruit_project/      # Core codebase
│   ├── main.py             # Main training script
│   ├── models/             # Model-related modules
│   │   ├── model_factory.py      # Factory for creating models
│   │   └── transforms_factory.py # Factory for creating data augmentations
│   └── utils/              # Utility modules
│       ├── data.py         # Data loading and processing
│       ├── early_stop.py   # Early stopping logic
│       ├── general.py      # General utility functions
│       ├── logging.py      # Logging utilities
│       ├── metrics.py      # Metrics and evaluation
│       └── trainer.py      # Training loop
├── pyproject.toml          # Build and tool configuration
└── README.md               # Project description
```

---

## Installation

This project supports both a local and a Docker-based setup. Choose the one that best fits your needs.

### 1. Local Setup (Without Docker)

First, clone the repository and navigate into the project directory:

```bash
git clone https://github.com/mohamedkhayat/fruit_internship.git
cd fruit_internship
```

It is highly recommended to use a virtual environment:
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### For Users (Training & Inference)
If you only want to use the project to train models or run inference, install the core package with the `torch` dependencies:
```bash
pip install .[torch]
```

#### For Developers (Contributing)
If you plan to contribute to the project, you will need the development dependencies (like `ruff`, `mypy`, etc.) in addition to the core package:
```bash
pip install .[dev,torch]
```

### 2. Docker Setup

The recommended way to work with this project is by using a containerized environment.

#### With VS Code (Recommended)
This is the easiest method, ensuring a consistent and reproducible environment.

1.  **Prerequisites:**
    *   [Docker Desktop](https://www.docker.com/products/docker-desktop/)
    *   [Visual Studio Code](https://code.visualstudio.com/)
    *   [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) for VS Code.

2.  **Launch:**
    *   Clone the repository.
    *   Open the project folder in VS Code.
    *   A notification will appear asking if you want to "Reopen in Container". Click it.

VS Code will automatically build the Docker image, install all necessary dependencies, and connect to the container. The environment will be ready for both using and developing the project.

#### Without VS Code (Manual Docker Build)
If you are not using VS Code, you can build and run the Docker container manually.

1.  **Prerequisites:**
    *   [Docker Desktop](https://www.docker.com/products/docker-desktop/)

2.  **Build the image:**
    From the project root directory, run:
    ```bash
    docker build -f .devcontainer/Dockerfile -t fruit-internship .
    ```

3.  **Run the container:**
    This command starts an interactive session inside the container, mounts your local project directory, and enables GPU access.

    *On Windows (Command Prompt/PowerShell):*
    ```bash
    docker run --gpus all --ipc=host -v "%cd%:/workspace" -it fruit-internship bash
    ```

    *On macOS/Linux:*
    ```bash
    docker run --gpus all --ipc=host -v "$(pwd):/workspace" -it fruit-internship bash
    ```

4.  **Install dependencies inside the container:**
    Once you have a shell inside the container, install the project dependencies:
    ```bash
    pip install --upgrade pip && pip install -e '.[dev]'
    ```
The environment is now ready.


---

## Dataset Setup

This project uses YOLO-format datasets with configurable folder structures. Each dataset requires a `data.yaml` configuration file.

### Supported Folder Structures

The project supports two common dataset layouts:

#### Structure A: Type-First (`type_first`) - Default
```
data/my_dataset/
├── data.yaml
├── images/
│   ├── train/
│   │   ├── img001.jpg
│   │   └── ...
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    │   ├── img001.txt
    │   └── ...
    ├── val/
    └── test/
```

#### Structure B: Split-First (`split_first`)
```
data/my_dataset/
├── data.yaml
├── train/
│   ├── images/
│   │   ├── img001.jpg
│   │   └── ...
│   └── labels/
│       ├── img001.txt
│       └── ...
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### Creating the Dataset Configuration File

Create a `data.yaml` file in your dataset root directory:

```yaml
# Folder structure configuration
folders:
  images: images          # Name of the images folder
  labels: labels          # Name of the labels folder
  train: train            # Name of the training split folder
  val: val                # Name of the validation split folder (use 'valid' if needed)
  test: test              # Name of the test split folder
  structure: type_first   # Either 'type_first' or 'split_first'

# Class configuration
nc: 3                     # Number of classes
names:                    # List of class names (order matters - matches label IDs)
  - Apple
  - Orange
  - Banana
```

### Label Format (YOLO)

Each image should have a corresponding `.txt` label file with the same name. Each line in the label file represents one object:

```
<class_id> <x_center> <y_center> <width> <height>
```

- `class_id`: Integer class index (0-indexed, matching the order in `names`)
- `x_center`, `y_center`: Normalized center coordinates (0-1)
- `width`, `height`: Normalized dimensions (0-1)

**Example label file (`img001.txt`):**
```
0 0.5 0.5 0.2 0.3
1 0.3 0.7 0.15 0.25
```

### Example: Setting Up a Roboflow Dataset

Datasets exported from Roboflow typically use the `split_first` structure:

```yaml
# data.yaml for Roboflow export
folders:
  images: images
  labels: labels
  train: train
  val: valid              # Roboflow uses 'valid' instead of 'val'
  test: test
  structure: split_first  # Roboflow uses split-first structure

nc: 12
names: ['Class1', 'Class2', 'Class3', ...]
```

### Running on Your Dataset

1. Place your dataset in the `data/` directory
2. Create or verify the `data.yaml` configuration file
3. Run training with your dataset:

```bash
python src/fruit_project/main.py root_dir=my_dataset data_conf_file=data.yaml
```

---

## Usage

### Train with Default Settings

```bash
python src/fruit_project/main.py
```

### Customize with Hydra CLI

```bash
python src/fruit_project/main.py model=rtdetrv2_50 lr=5e-5 aug=safe
```

### Common Training Examples

**Train on a custom dataset:**
```bash
python src/fruit_project/main.py root_dir=my_dataset data_conf_file=data.yaml
```

**Train with a larger model and more epochs:**
```bash
python src/fruit_project/main.py \
    model=dfine_xlarge_obj365 \
    root_dir=my_dataset \
    data_conf_file=data.yaml \
    epochs=80 \
    lr=1e-4 \
    effective_batch_size=64 \
    step_batch_size=16
```

**Train with Mosaic augmentation:**
```bash
python src/fruit_project/main.py \
    root_dir=my_dataset \
    mosaic.use=True \
    mosaic.prob=0.4 \
    aug=hard
```

**Train with mixed precision and EMA:**
```bash
python src/fruit_project/main.py \
    root_dir=my_dataset \
    fp16=True \
    ema.use=True \
    ema.decay=0.999
```

---

## Configuration

This project uses two types of configuration files:

### 1. Project Configuration (`conf/config.yaml`)

The main training configuration file using Hydra. This controls model selection, training hyperparameters, and runtime options.

**Key Training Parameters:**

| Parameter | Description | Default |
|---|---|---|
| `model` | Model configuration to use (from `conf/model/`). | `rtdetrv2_50` |
| `effective_batch_size` | The total batch size (used for gradient accumulation). | `64` |
| `step_batch_size` | The batch size per forward pass. | `4` |
| `epochs` | The total number of training epochs. | `30` |
| `lr` | The base learning rate. | `1e-4` |
| `weight_decay` | The weight decay for the optimizer. | `1e-4` |
| `warmup_epochs` | The number of warmup epochs for the LR scheduler. | `5` |
| `patience` | The patience for early stopping. | `15` |
| `delta` | Minimum improvement for early stopping. | `0.001` |

**Dataset Parameters:**

| Parameter | Description | Default |
|---|---|---|
| `root_dir` | The dataset directory name (under `data/`). | `Fruit_dataset` |
| `data_conf_file` | The dataset configuration YAML file name. | `data.yaml` |

**Learning Rate Scaling:**

| Parameter | Description | Default |
|---|---|---|
| `lr_back_factor` | Factor to divide `lr` by for the backbone. | `10` |
| `lr_neck_factor` | Factor to divide `lr` by for the neck/encoder. | `5` |
| `smart_optim` | Use smart parameter grouping for optimizer. | `False` |

**Augmentation & Sampling:**

| Parameter | Description | Default |
|---|---|---|
| `aug` | The augmentation level (`hard` or `safe`). | `hard` |
| `do_sample` | Whether to use weighted random sampling. | `False` |
| `min_area` | Minimum bounding box area to keep. | `15.0` |

**Mosaic Augmentation:**

| Parameter | Description | Default |
|---|---|---|
| `mosaic.use` | Whether to use Mosaic augmentation. | `False` |
| `mosaic.prob` | The probability of applying Mosaic. | `0.0` |
| `mosaic.disable_epoch` | Epochs before end to disable Mosaic. | `15` |

**Model Training Options:**

| Parameter | Description | Default |
|---|---|---|
| `freeze_backbone` | Whether to freeze the backbone. | `False` |
| `partially_freeze_backbone` | Unfreeze last stage of backbone. | `False` |
| `fp16` | Use mixed precision (FP16) training. | `True` |
| `optim` | Optimizer type (`torch`, `8bit`). | `8bit` |

**EMA (Exponential Moving Average):**

| Parameter | Description | Default |
|---|---|---|
| `ema.use` | Whether to use EMA weights. | `True` |
| `ema.decay` | EMA decay factor. | `0.999` |

**Checkpointing:**

| Parameter | Description | Default |
|---|---|---|
| `ckpt.save` | Whether to save checkpoints. | `False` |
| `ckpt.load.model_only` | Load only model weights from checkpoint. | `False` |
| `ckpt.load.all` | Load full training state from checkpoint. | `False` |

**Logging & Misc:**

| Parameter | Description | Default |
|---|---|---|
| `log` | Whether to log to Weights & Biases. | `True` |
| `seed` | The random seed for reproducibility. | `42` |
| `num_workers` | Number of dataloader workers. | `4` |
| `n_images` | Number of images to log for visualization. | `6` |
| `upload` | Whether to upload model artifacts. | `False` |

### 2. Dataset Configuration (`data/<dataset>/data.yaml`)

Each dataset has its own configuration file that defines the folder structure and class labels. See the [Dataset Setup](#dataset-setup) section for details.

---

## Features

* Transformer-based object detection (RT-DETRv2)
* Modular model factory (configurable via YAML)
* Differentiable learning rates for fine-tuning (backbone, encoder/decoder, prediction heads)
* Gradient accumulation to simulate larger batch sizes
* Advanced augmentation pipelines with Albumentations, including Mosaic
* Stratified sampling (max/mean) to handle class imbalance
* Full integration with Weights & Biases:

  * Detailed metric logging: mAP, mAP@50, Precision, Recall, and loss components
  * Bounding box visualizations
  * Class distribution and confusion matrix plots
  * Checkpoint artifact logging

* Mixed precision support using `torch.cuda.amp`
* Early stopping with automatic best model restoration
* Clean and extensible codebase

---

## Documentation

Auto-generated using Sphinx + AutoAPI.

view docs:
`https://mohamedkhayat.github.io/fruit_internship`

---

## Roadmap

* Support multi-task training (detection + regression)
* Unit test coverage and CI for training and data loaders

---

## License

This project is licensed under the AGPL‑3.0‑or‑later license. See [LICENSE](LICENSE) for details.

If you deploy this code (e.g., via a web service), you must make the full source code available to your users, per AGPL §13.
