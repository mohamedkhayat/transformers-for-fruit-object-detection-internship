# Object Detection Framework

A flexible, production-ready object detection framework built on Hugging Face Transformers. Works with **any YOLO-format dataset** and supports a wide range of transformer-based detection models out of the box.

> Originally developed as part of a university internship for fruit localization, this framework has evolved into a general-purpose object detection pipeline suitable for any detection task.

---

## Overview

This repository provides a modular and reproducible deep learning pipeline for object detection using transformer models from Hugging Face. It is built with scalability and maintainability in mind, following best practices in code structure, configuration, and logging.

**Use Cases:**
- Custom object detection for any domain (agriculture, medical, industrial, etc.)
- Fine-tuning pretrained models on your own datasets
- Rapid experimentation with different model architectures
- Production training with advanced features (EMA, mixed precision, gradient accumulation)

Key components:
- **25+ pretrained models** including RT-DETR, D-FINE, DETR, YOLOS, and more
- Configurable training via Hydra
- Advanced augmentations with Albumentations (including Mosaic)
- Experiment tracking with Weights & Biases
- Early stopping and mixed precision training
- Class-imbalanced sampling strategies
- Flexible dataset configuration (supports multiple folder structures)
- Sphinx-based auto-generated documentation

---

## Directory Structure

```
mohamedkhayat-fruit_internship/
├── conf/                   # Hydra configuration files
│   ├── config.yaml         # Base training config
│   └── model/              # Model-specific configs
├── checkpoints/            # Model checkpoints
├── docs/                   # Sphinx documentation
│   ├── conf.py             # Sphinx configuration
│   └── index.rst           # Main documentation file
├── src/fruit_project/      # Core codebase
│   ├── main.py             # Main training script
│   ├── config.py           # Shared configuration for inference (models, classes, etc.)
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
├── streamlit_app.py        # Inference UI application
├── upload_to_hf.py         # Script to upload models to HuggingFace
├── pyproject.toml          # Build and tool configuration
└── README.md               # Project description
```

---

## Supported Models

The framework supports **25+ pretrained models** from Hugging Face, organized by architecture:

| Model Family | Variants | Description |
|---|---|---|
| **RT-DETR v2** | `rtdetrv2_18`, `rtdetrv2_34`, `rtdetrv2_50`, `rtdetrv2_101` | Real-Time DETR with improved accuracy |
| **RT-DETR v1** | `rtdetrv1_18`, `rtdetrv1_34`, `rtdetrv1_50`, `rtdetrv1_101` | Original Real-Time DETR |
| **RT-DETR v1 (Objects365)** | `rtdetrv1_50_365`, `rtdetrv1_101_365` | Pretrained on Objects365 dataset |
| **D-FINE** | `dfine_large_coco`, `dfine_xlarge_coco` | High-accuracy detection model |
| **D-FINE (Objects365)** | `dfine_large_obj365`, `dfine_xlarge_obj365` | Pretrained on Objects365 |
| **D-FINE (Obj2COCO)** | `dfine_large_obj2coco`, `dfine_xlarge_obj2coco` | Objects365 → COCO transfer |
| **DETR** | `detr_50`, `detr_101`, `detr_50_dc5` | Original Facebook DETR |
| **Conditional DETR** | `cond_detr_50` | Faster converging DETR variant |
| **DAB-DETR** | `dab_detr_50` | Dynamic anchor boxes DETR |
| **Deformable DETR** | `defor_detr` | Deformable attention DETR |
| **YOLOS** | `yolos_tiny`, `yolos_small`, `yolos_base` | ViT-based detection |

### Adding a New Model

To add support for a new Hugging Face model:

#### Step 1: Create a Model Configuration File

Create a new YAML file in `conf/model/` (e.g., `conf/model/my_new_model.yaml`):

```yaml
name: my_new_model          # Unique identifier (must match filename)
input_height: 640           # Model input height
input_width: 640            # Model input width
do_normalize: True          # Whether to normalize images (check model docs)
grad_max_norm: 0.1          # Gradient clipping max norm
```

#### Step 2: Register the Model in the Config

Add your model to the `SUPPORTED_MODELS` dictionary in `src/fruit_project/config.py`:

```python
SUPPORTED_MODELS: Dict[str, str] = {
    # ... existing models ...
    "my_new_model": "huggingface-org/model-checkpoint-name",
}
```

#### Step 3: Use Your Model

```bash
python src/fruit_project/main.py model=my_new_model root_dir=my_dataset
```

### Model Configuration Options

| Parameter | Description | Typical Values |
|---|---|---|
| `name` | Unique model identifier (must match `supported_models` key) | String |
| `input_height` | Input image height in pixels | `640`, `800`, `1024` |
| `input_width` | Input image width in pixels | `640`, `800`, `1024` |
| `do_normalize` | Whether the processor should normalize images | `True` / `False` |
| `grad_max_norm` | Maximum gradient norm for clipping | `0.1` (typical for DETR-like models) |

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

* **25+ pretrained models** - RT-DETR, D-FINE, DETR, YOLOS, and more from Hugging Face
* **Any YOLO-format dataset** - Flexible folder structure configuration
* **Easy model extension** - Add new models with just a YAML config file
* Modular model factory (configurable via YAML)
* Differentiable learning rates for fine-tuning (backbone, neck, prediction heads)
* Gradient accumulation to simulate larger batch sizes
* Advanced augmentation pipelines with Albumentations, including Mosaic
* Stratified sampling (max/mean) to handle class imbalance
* Full integration with Weights & Biases:

  * Detailed metric logging: mAP, mAP@50, Precision, Recall, and loss components
  * Bounding box visualizations
  * Class distribution and confusion matrix plots
  * Checkpoint artifact logging

* Mixed precision support using `torch.cuda.amp`
* Exponential Moving Average (EMA) for stable training
* Early stopping with automatic best model restoration
* Clean and extensible codebase

---

## Inference

### Streamlit UI

A web-based interface for running inference with trained models. Supports both local checkpoints and HuggingFace-hosted models.

**Install Streamlit:**
```bash
pip install streamlit
```

**Run the app:**
```bash
streamlit run streamlit_app.py
```

The UI allows you to:
1. **Select Model Source**: Choose between `local` (`.pth` files from `checkpoints/save/`) or `huggingface` (pre-trained models)
2. **Select Model**: Pick from available models based on your source selection
3. **Upload Image**: Upload any image containing fruits/vegetables
4. **Adjust Threshold**: Set the confidence threshold for detections
5. **Run Detection**: View bounding boxes and class predictions

### Pre-trained Models on HuggingFace

We provide **16 pre-trained fruit detection models** on HuggingFace Hub:

| Model | Architecture | mAP@50 | HuggingFace ID |
|-------|-------------|--------|----------------|
| RT-DETRv2-101 | RT-DETR v2 | 0.6797 | `MohamedKhayat/fruit-detector-rtdetrv2-101` |
| RT-DETRv1-101 | RT-DETR v1 | 0.6775 | `MohamedKhayat/fruit-detector-rtdetrv1-101` |
| RT-DETRv1-50 | RT-DETR v1 | 0.6678 | `MohamedKhayat/fruit-detector-rtdetrv1-50` |
| D-FINE XLarge | D-FINE | 0.6631 | `MohamedKhayat/fruit-detector-dfine-xlarge` |
| RT-DETRv1-50 (obj365) | RT-DETR v1 | 0.6524 | `MohamedKhayat/fruit-detector-rtdetrv1-50-obj365` |
| D-FINE Large | D-FINE | 0.6524 | `MohamedKhayat/fruit-detector-dfine-large` |
| RT-DETRv2-50 | RT-DETR v2 | 0.6506 | `MohamedKhayat/fruit-detector-rtdetrv2-50` |
| D-FINE Large (obj365) | D-FINE | 0.6448 | `MohamedKhayat/fruit-detector-dfine-large-obj365` |
| RT-DETRv1-101 (obj365) | RT-DETR v1 | 0.6299 | `MohamedKhayat/fruit-detector-rtdetrv1-101-obj365` |
| D-FINE XLarge (obj365) | D-FINE | 0.6085 | `MohamedKhayat/fruit-detector-dfine-xlarge-obj365` |
| Deformable DETR | Deformable DETR | 0.5961 | `MohamedKhayat/fruit-detector-deformable-detr` |
| Conditional DETR-50 | Conditional DETR | 0.5820 | `MohamedKhayat/fruit-detector-conditional-detr-50` |
| DETR-101 | DETR | 0.5712 | `MohamedKhayat/fruit-detector-detr-101` |
| DAB-DETR-50 | DAB-DETR | 0.5695 | `MohamedKhayat/fruit-detector-dab-detr-50` |
| DETR-50 | DETR | 0.5694 | `MohamedKhayat/fruit-detector-detr-50` |
| YOLOS Base | YOLOS | 0.5585 | `MohamedKhayat/fruit-detector-yolos-base` |

**Detected Classes (12):** Apple, Cherry, Figs, Olive, Pomegranate, Orange, Rockmelon, Strawberry, Potato, Tomato, Watermelon, Bell-pepper

### Python Usage

```python
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image
import torch

# Load model and processor
model_id = "MohamedKhayat/fruit-detector-rtdetrv2-101"
processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForObjectDetection.from_pretrained(model_id)

# Load and process image
image = Image.open("fruit_image.jpg")
inputs = processor(images=image, return_tensors="pt")

# Run inference
with torch.no_grad():
    outputs = model(**inputs)

# Post-process results
target_sizes = torch.tensor([[image.height, image.width]])
results = processor.post_process_object_detection(
    outputs, threshold=0.5, target_sizes=target_sizes
)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    print(f"Detected {model.config.id2label[label.item()]} with confidence {score:.2f}")
```

### Uploading Models to HuggingFace

To upload your trained checkpoints to HuggingFace Hub:

```bash
export HF_TOKEN="your_huggingface_token"
python upload_to_hf.py \
    --checkpoint checkpoints/save/model-rtdetrv2_50_lr-0.0001_0902_2210_0.6506.pth \
    --repo-name fruit-detector-my-model
```

---

## Documentation

Auto-generated using Sphinx + AutoAPI.

view docs:
`https://mohamedkhayat.github.io/fruit_internship`

---

## License

This project is licensed under the AGPL‑3.0‑or‑later license. See [LICENSE](LICENSE) for details.

If you deploy this code (e.g., via a web service), you must make the full source code available to your users, per AGPL §13.
