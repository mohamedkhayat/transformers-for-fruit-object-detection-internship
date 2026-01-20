# SPDX-FileCopyrightText: 2025 Mohamed Khayat
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Shared configuration for fruit detection models.
This module provides a single source of truth for model configs, class labels, and related settings.
"""

from pathlib import Path
from typing import Dict, List, Optional
import yaml

_THIS_DIR = Path(__file__).parent
_WORKSPACE_ROOT = _THIS_DIR.parent if _THIS_DIR.name == "src" else _THIS_DIR

CONF_DIR = _WORKSPACE_ROOT / "conf"
MODEL_CONF_DIR = CONF_DIR / "model"
DATA_YAML_PATH = _WORKSPACE_ROOT / "data" / "Fruit_dataset" / "data.yaml"
CHECKPOINT_DIR = _WORKSPACE_ROOT / "checkpoints" / "save"

SUPPORTED_MODELS: Dict[str, str] = {
    "rtdetrv2_18": "PekingU/rtdetr_v2_r18vd",
    "rtdetrv2_34": "PekingU/rtdetr_v2_r34vd",
    "rtdetrv2_50": "PekingU/rtdetr_v2_r50vd",
    "rtdetrv2_101": "PekingU/rtdetr_v2_r101vd",
    "rtdetrv1_18": "PekingU/rtdetr_r18vd",
    "rtdetrv1_34": "PekingU/rtdetr_r34vd",
    "rtdetrv1_50": "PekingU/rtdetr_r50vd",
    "rtdetrv1_50_365": "PekingU/rtdetr_r50vd_coco_o365",
    "rtdetrv1_101": "PekingU/rtdetr_r101vd",
    "rtdetrv1_101_365": "PekingU/rtdetr_r101vd_coco_o365",
    "detr_50": "facebook/detr-resnet-50",
    "detr_101": "facebook/detr-resnet-101",
    "detr_50_dc5": "facebook/detr-resnet-50-dc5",
    "cond_detr_50": "microsoft/conditional-detr-resnet-50",
    "yolos_tiny": "hustvl/yolos-tiny",
    "yolos_small": "hustvl/yolos-small",
    "yolos_base": "hustvl/yolos-base",
    "defor_detr": "SenseTime/deformable-detr",
    "dab_detr_50": "IDEA-Research/dab-detr-resnet-50",
    "dfine_large_coco": "ustc-community/dfine-large-coco",
    "dfine_xlarge_coco": "ustc-community/dfine-xlarge-coco",
    "dfine_large_obj365": "ustc-community/dfine-large-obj365",
    "dfine_xlarge_obj365": "ustc-community/dfine-xlarge-obj365",
    "dfine_large_obj2coco": "ustc-community/dfine-large-obj2coco-e25",
    "dfine_xlarge_obj2coco": "ustc-community/dfine-xlarge-obj2coco",
}

CLASS_NAMES: List[str] = [
    "Apple",
    "Cherry",
    "Figs",
    "Olive",
    "Pomegranate",
    "Orange",
    "Rockmelon",
    "Strawberry",
    "Potato",
    "Tomato",
    "Watermelon",
    "Bell-pepper",
]

NUM_CLASSES: int = len(CLASS_NAMES)
ID2LABEL: Dict[int, str] = {i: name for i, name in enumerate(CLASS_NAMES)}
LABEL2ID: Dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}

COLORS: List[str] = [
    "#FF6B6B",  # Apple - Red
    "#C0392B",  # Cherry - Dark Red
    "#8E44AD",  # Figs - Purple
    "#2ECC71",  # Olive - Green
    "#E74C3C",  # Pomegranate - Crimson
    "#F39C12",  # Orange - Orange
    "#F1C40F",  # Rockmelon - Yellow
    "#E91E63",  # Strawberry - Pink
    "#795548",  # Potato - Brown
    "#FF5722",  # Tomato - Deep Orange
    "#4CAF50",  # Watermelon - Light Green
    "#FFEB3B",  # Bell-pepper - Yellow-Green
]


def get_model_config(model_name: str) -> Dict:
    """
    Load model configuration from YAML file.

    Args:
        model_name: Name of the model (e.g., 'rtdetrv2_50')

    Returns:
        Dictionary with model configuration
    """
    yaml_path = MODEL_CONF_DIR / f"{model_name}.yaml"
    if yaml_path.exists():
        with open(yaml_path) as f:
            return yaml.safe_load(f)

    return {
        "name": model_name,
        "input_height": 640,
        "input_width": 640,
        "do_normalize": not model_name.startswith("rtdetrv2"),
        "grad_max_norm": 0.1,
    }


def get_do_normalize(model_name: str) -> bool:
    """
    Get whether a model requires image normalization.

    Args:
        model_name: Name of the model

    Returns:
        True if model requires normalization
    """
    config = get_model_config(model_name)
    return config.get("do_normalize", True)


def get_hf_checkpoint(model_name: str) -> Optional[str]:
    """
    Get the HuggingFace checkpoint ID for a model.

    Args:
        model_name: Name of the model

    Returns:
        HuggingFace checkpoint string or None
    """
    return SUPPORTED_MODELS.get(model_name)


def get_input_size(model_name: str) -> tuple:
    """
    Get the input size for a model.

    Args:
        model_name: Name of the model

    Returns:
        Tuple of (height, width)
    """
    config = get_model_config(model_name)
    return (config.get("input_height", 640), config.get("input_width", 640))


HF_FRUIT_MODELS: Dict[str, str] = {
    # RT-DETR v2
    "MohamedKhayat/fruit-detector-rtdetrv2-50": "rtdetrv2_50",
    "MohamedKhayat/fruit-detector-rtdetrv2-101": "rtdetrv2_101",
    # RT-DETR v1
    "MohamedKhayat/fruit-detector-rtdetrv1-50": "rtdetrv1_50",
    "MohamedKhayat/fruit-detector-rtdetrv1-50-obj365": "rtdetrv1_50_365",
    "MohamedKhayat/fruit-detector-rtdetrv1-101": "rtdetrv1_101",
    "MohamedKhayat/fruit-detector-rtdetrv1-101-obj365": "rtdetrv1_101_365",
    # DETR
    "MohamedKhayat/fruit-detector-detr-50": "detr_50",
    "MohamedKhayat/fruit-detector-detr-101": "detr_101",
    "MohamedKhayat/fruit-detector-conditional-detr-50": "cond_detr_50",
    "MohamedKhayat/fruit-detector-deformable-detr": "defor_detr",
    "MohamedKhayat/fruit-detector-dab-detr-50": "dab_detr_50",
    # D-FINE
    "MohamedKhayat/fruit-detector-dfine-large": "dfine_large_coco",
    "MohamedKhayat/fruit-detector-dfine-large-obj365": "dfine_large_obj365",
    "MohamedKhayat/fruit-detector-dfine-xlarge": "dfine_xlarge_coco",
    "MohamedKhayat/fruit-detector-dfine-xlarge-obj365": "dfine_xlarge_obj365",
    # YOLOS
    "MohamedKhayat/fruit-detector-yolos-base": "yolos_base",
}


def load_class_names_from_yaml(yaml_path: Optional[Path] = None) -> List[str]:
    """
    Load class names from data.yaml file.

    Args:
        yaml_path: Path to the yaml file (defaults to DATA_YAML_PATH)

    Returns:
        List of class names
    """
    path = yaml_path or DATA_YAML_PATH
    if not path.exists():
        return CLASS_NAMES

    with open(path) as f:
        data = yaml.safe_load(f)

    return data.get("names", CLASS_NAMES)
