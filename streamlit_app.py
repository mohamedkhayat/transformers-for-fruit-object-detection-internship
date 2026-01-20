# SPDX-FileCopyrightText: 2025 Mohamed Khayat
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Streamlit UI for Fruit Detection using trained models.
Run with: streamlit run streamlit_app.py
"""

import sys
from pathlib import Path

# Add src to path to import fruit_project modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

import os
import glob
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum

import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor, AutoModelForObjectDetection, AutoConfig
import albumentations as A

# Import shared configuration
from fruit_project.config import (
    SUPPORTED_MODELS,
    CLASS_NAMES,
    NUM_CLASSES,
    ID2LABEL,
    LABEL2ID,
    COLORS,
    CHECKPOINT_DIR,
    HF_FRUIT_MODELS,
    get_do_normalize,
)


# ============================================================================
# DATA CLASSES
# ============================================================================


class ModelSource(Enum):
    """Source type for model loading."""

    LOCAL = "local"
    HUGGINGFACE = "huggingface"


@dataclass
class ModelInfo:
    """Information about a model checkpoint."""

    display_name: str
    source: ModelSource
    path_or_id: str
    architecture: str
    score: Optional[str] = None


@dataclass
class Detection:
    """A single object detection result."""

    box: List[float]
    label: int
    label_name: str
    score: float


# ============================================================================
# MODEL DISCOVERY
# ============================================================================


def discover_local_checkpoints(
    checkpoint_dir: str = str(CHECKPOINT_DIR),
) -> List[ModelInfo]:
    """
    Scan the checkpoint directory for available local models.

    Args:
        checkpoint_dir: Directory containing .pth checkpoint files

    Returns:
        List of ModelInfo for local checkpoints
    """
    models = []
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))

    for ckpt_path in checkpoint_files:
        filename = os.path.basename(ckpt_path)
        # Parse: model-{arch}_lr-{lr}_{date}_{time}_{score}.pth
        match = re.match(r"model-(.+?)_lr-[\d.]+_\d+_\d+_([\d.]+)\.pth", filename)
        if match:
            arch = match.group(1)
            score = match.group(2)
            models.append(
                ModelInfo(
                    display_name=f"[Local] {arch} (mAP: {score})",
                    source=ModelSource.LOCAL,
                    path_or_id=ckpt_path,
                    architecture=arch,
                    score=score,
                )
            )

    return sorted(models, key=lambda m: (m.architecture, m.score or ""), reverse=True)


def discover_hf_models() -> List[ModelInfo]:
    """
    Get list of HuggingFace hosted fruit detection models.

    Returns:
        List of ModelInfo for HuggingFace models
    """
    models = []
    for hf_id, arch in HF_FRUIT_MODELS.items():
        models.append(
            ModelInfo(
                display_name=f"[HuggingFace] {hf_id.split('/')[-1]}",
                source=ModelSource.HUGGINGFACE,
                path_or_id=hf_id,
                architecture=arch,
            )
        )
    return models


def get_all_available_models() -> Dict[str, ModelInfo]:
    """
    Get all available models from both local checkpoints and HuggingFace.

    Returns:
        Dict mapping display name to ModelInfo
    """
    models = {}

    # Local checkpoints
    for model in discover_local_checkpoints():
        models[model.display_name] = model

    # HuggingFace models
    for model in discover_hf_models():
        models[model.display_name] = model

    return models


# ============================================================================
# MODEL LOADING
# ============================================================================


@st.cache_resource
def load_model_from_local(checkpoint_path: str, model_name: str, device: str):
    """
    Load a model from a local .pth checkpoint.

    Args:
        checkpoint_path: Path to the .pth checkpoint file
        model_name: Name of the model architecture
        device: Device to load the model on

    Returns:
        Tuple of (model, processor, id2label)
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Model {model_name} not supported")

    hf_checkpoint = SUPPORTED_MODELS[model_name]
    do_normalize = get_do_normalize(model_name)

    # Load config with our labels
    config = AutoConfig.from_pretrained(
        hf_checkpoint,
        trust_remote_code=True,
        num_labels=NUM_CLASSES,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Load processor
    processor = AutoImageProcessor.from_pretrained(
        hf_checkpoint,
        trust_remote_code=True,
        use_fast=True,
        do_resize=True,
        do_pad=True,
        do_normalize=do_normalize,
        size={"max_height": 640, "max_width": 640},
        pad_size={"height": 640, "width": 640},
    )

    # Create model from config (no weight download)
    model_kwargs = {}
    if "yolos" in model_name:
        model_kwargs.update(
            {
                "attn_implementation": "sdpa",
                "torch_dtype": torch.float32,
            }
        )

    model = AutoModelForObjectDetection.from_config(config, **model_kwargs)

    # Load trained weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    return model, processor, id2label


@st.cache_resource
def load_model_from_huggingface(hf_model_id: str, device: str):
    """
    Load a model directly from HuggingFace Hub.

    Args:
        hf_model_id: HuggingFace model repository ID
        device: Device to load the model on

    Returns:
        Tuple of (model, processor, id2label)
    """
    # Load processor
    processor = AutoImageProcessor.from_pretrained(
        hf_model_id,
        trust_remote_code=True,
        use_fast=True,
    )

    # Load model with weights
    model = AutoModelForObjectDetection.from_pretrained(
        hf_model_id,
        trust_remote_code=True,
    )

    model.to(device)
    model.eval()

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    return model, processor, id2label


def load_model(model_info: ModelInfo, device: str):
    """
    Load a model based on its source.

    Args:
        model_info: ModelInfo object describing the model
        device: Device to load the model on

    Returns:
        Tuple of (model, processor, id2label)
    """
    if model_info.source == ModelSource.LOCAL:
        return load_model_from_local(
            model_info.path_or_id, model_info.architecture, device
        )
    else:
        return load_model_from_huggingface(model_info.path_or_id, device)


# ============================================================================
# INFERENCE
# ============================================================================


def get_inference_transforms(height: int = 640, width: int = 640) -> A.Compose:
    """Get transforms for inference matching training test transforms."""
    return A.Compose(
        [
            A.SmallestMaxSize(max_size_hw=(height, width), p=1.0),
            A.CenterCrop(height=height, width=width, p=1.0),
        ]
    )


def run_inference(
    model,
    processor,
    image: Image.Image,
    device: str,
    id2label: Dict[int, str],
    confidence_threshold: float = 0.5,
) -> Tuple[List[Detection], Image.Image]:
    """
    Run object detection inference on an image.

    Args:
        model: The detection model
        processor: The image processor
        image: PIL Image to process
        device: Device to run inference on
        id2label: Mapping from label IDs to label names
        confidence_threshold: Minimum confidence threshold

    Returns:
        Tuple of (list of Detection objects, transformed image)
    """
    # Apply transforms
    transforms = get_inference_transforms(640, 640)
    img_array = np.array(image)
    transformed = transforms(image=img_array)
    transformed_image = Image.fromarray(transformed["image"])

    # Process image
    inputs = processor(images=transformed_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process
    target_sizes = torch.tensor(
        [[transformed_image.height, transformed_image.width]], device=device
    )
    results = processor.post_process_object_detection(
        outputs,
        threshold=confidence_threshold,
        target_sizes=target_sizes,
    )[0]

    # Convert to Detection objects
    detections = []
    for score, label, box in zip(
        results["scores"].cpu().numpy(),
        results["labels"].cpu().numpy(),
        results["boxes"].cpu().numpy(),
    ):
        label_id = int(label)
        # Skip background class if present
        if label_id >= NUM_CLASSES:
            continue

        detections.append(
            Detection(
                box=box.tolist(),
                label=label_id,
                label_name=id2label.get(label_id, f"Unknown({label_id})"),
                score=float(score),
            )
        )

    return detections, transformed_image


# ============================================================================
# VISUALIZATION
# ============================================================================


def get_font(font_size: int) -> ImageFont.FreeTypeFont:
    """Load a font, with fallbacks."""
    font_paths = [
        "fonts/FiraCodeNerdFont-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, font_size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def draw_detections(
    image: Image.Image,
    detections: List[Detection],
    line_width: int = 3,
    font_size: int = 16,
) -> Image.Image:
    """Draw bounding boxes and labels on an image."""
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    font = get_font(font_size)

    for det in detections:
        color = COLORS[det.label % len(COLORS)]
        x1, y1, x2, y2 = det.box

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # Draw label
        label_text = f"{det.label_name}: {det.score:.2f}"
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        padding = 4
        label_y = max(y1 - text_height - padding * 2, 0)

        draw.rectangle(
            [
                x1,
                label_y,
                x1 + text_width + padding * 2,
                label_y + text_height + padding * 2,
            ],
            fill=color,
        )
        draw.text(
            (x1 + padding, label_y + padding), label_text, fill="white", font=font
        )

    return img_draw


# ============================================================================
# UI COMPONENTS
# ============================================================================


def render_sidebar() -> Tuple[Optional[ModelInfo], float, int, int, str]:
    """Render the sidebar and return user selections."""
    with st.sidebar:
        st.header("⚙️ Settings")

        # Model selection
        models = get_all_available_models()

        if not models:
            st.error("No models found. Add local checkpoints or configure HF models.")
            st.stop()

        selected_name = st.selectbox(
            "Select Model",
            options=list(models.keys()),
            help="Choose a model for detection",
        )

        model_info = models[selected_name]

        # Model info
        st.caption(f"Architecture: `{model_info.architecture}`")
        st.caption(f"Source: `{model_info.source.value}`")
        if model_info.source == ModelSource.LOCAL:
            st.caption(
                f"Base: `{SUPPORTED_MODELS.get(model_info.architecture, 'Unknown')}`"
            )

        st.divider()

        # Detection settings
        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
        )

        # Display settings
        st.subheader("Display")
        line_width = st.slider("Box Line Width", 1, 10, 3)
        font_size = st.slider("Font Size", 10, 30, 16)

        # Device info
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Device: **{device.upper()}**")

        st.divider()

        # Class legend
        st.subheader("Classes")
        for i, (name, color) in enumerate(zip(CLASS_NAMES, COLORS)):
            st.markdown(
                f'<span style="color:{color}; font-weight:bold;">●</span> {name}',
                unsafe_allow_html=True,
            )

        return model_info, confidence, line_width, font_size, device


def render_main_content(
    model_info: ModelInfo,
    confidence: float,
    line_width: int,
    font_size: int,
    device: str,
):
    """Render the main content area."""
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png", "webp"],
            key="image_uploader",
        )

        image = None
        if uploaded_file is not None:
            # Clear results on new image
            current_id = uploaded_file.file_id
            if st.session_state.get("last_file_id") != current_id:
                st.session_state["last_file_id"] = current_id
                for key in ["detections", "result_image"]:
                    st.session_state.pop(key, None)

            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.subheader("🔍 Results")

        if image is not None:
            if st.button("🚀 Run Detection", type="primary", use_container_width=True):
                with st.spinner("Loading model..."):
                    try:
                        model, processor, id2label = load_model(model_info, device)
                    except Exception as e:
                        st.error(f"Failed to load model: {e}")
                        st.stop()

                with st.spinner("Running inference..."):
                    try:
                        detections, transformed = run_inference(
                            model, processor, image, device, id2label, confidence
                        )
                    except Exception as e:
                        st.error(f"Inference failed: {e}")
                        st.stop()

                st.session_state["detections"] = detections
                st.session_state["result_image"] = draw_detections(
                    transformed, detections, line_width, font_size
                )

            # Display results
            if "result_image" in st.session_state:
                st.image(
                    st.session_state["result_image"],
                    caption="Detection Results",
                    use_container_width=True,
                )

                detections = st.session_state.get("detections", [])
                if detections:
                    st.success(f"Found {len(detections)} object(s)")

                    # Detection table
                    st.subheader("📊 Details")
                    data = [
                        {
                            "#": i,
                            "Class": d.label_name,
                            "Confidence": f"{d.score:.2%}",
                            "Box": f"({d.box[0]:.0f}, {d.box[1]:.0f}, {d.box[2]:.0f}, {d.box[3]:.0f})",
                        }
                        for i, d in enumerate(detections, 1)
                    ]
                    st.dataframe(data, use_container_width=True)

                    # Summary
                    st.subheader("📈 Summary")
                    counts = {}
                    for d in detections:
                        counts[d.label_name] = counts.get(d.label_name, 0) + 1
                    for name, count in sorted(counts.items(), key=lambda x: -x[1]):
                        st.write(f"- **{name}**: {count}")
                else:
                    st.warning("No objects detected above threshold.")
        else:
            st.info("👆 Upload an image to get started")


# ============================================================================
# MAIN
# ============================================================================


def main():
    st.set_page_config(
        page_title="Fruit Detection",
        page_icon="🍎",
        layout="wide",
    )

    st.title("🍎 Fruit Detection")
    st.markdown("Upload an image and select a model to detect fruits.")

    # Render UI
    model_info, confidence, line_width, font_size, device = render_sidebar()
    render_main_content(model_info, confidence, line_width, font_size, device)


if __name__ == "__main__":
    main()
