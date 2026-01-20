# SPDX-FileCopyrightText: 2025 Mohamed Khayat
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Upload trained fruit detection models to HuggingFace Hub.

Usage:
    python upload_to_hf.py --checkpoint checkpoints/save/model-rtdetrv2_50_lr-0.0001_0902_2210_0.6506.pth \
                           --repo-name fruit-detector-rtdetrv2-50 \
                           --hf-token YOUR_HF_TOKEN

    # Or use environment variable for token:
    export HF_TOKEN="hf_your_token_here"
    python upload_to_hf.py --checkpoint checkpoints/save/model-rtdetrv2_50_lr-0.0001_0902_2210_0.6506.pth \
                           --repo-name fruit-detector-rtdetrv2-50
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import argparse
import os
import re
import shutil
import torch
from huggingface_hub import HfApi, create_repo
from transformers import AutoConfig, AutoImageProcessor, AutoModelForObjectDetection

from fruit_project.config import (
    SUPPORTED_MODELS,
    CLASS_NAMES,
    NUM_CLASSES,
    ID2LABEL,
    LABEL2ID,
    get_do_normalize,
)


def extract_model_name(checkpoint_path: str) -> str:
    """Extract model name from checkpoint filename."""
    filename = os.path.basename(checkpoint_path)
    match = re.match(r"model-(.+?)_lr-", filename)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract model name from: {filename}")


def extract_map_score(checkpoint_path: str) -> str:
    """Extract mAP score from checkpoint filename."""
    filename = os.path.basename(checkpoint_path)
    match = re.match(r"model-.+?_lr-[\d.]+_\d+_\d+_([\d.]+)\.pth", filename)
    if match:
        return match.group(1)
    return "unknown"


def load_and_prepare_model(checkpoint_path: str, model_name: str):
    """Load checkpoint and prepare model for upload."""
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Model {model_name} not supported. Available: {list(SUPPORTED_MODELS.keys())}"
        )

    hf_checkpoint = SUPPORTED_MODELS[model_name]
    do_normalize = get_do_normalize(model_name)

    print(f"  Loading config from: {hf_checkpoint}")

    config = AutoConfig.from_pretrained(
        hf_checkpoint,
        trust_remote_code=True,
        num_labels=NUM_CLASSES,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

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

    model_kwargs = {}
    if "yolos" in model_name:
        model_kwargs["attn_implementation"] = "sdpa"

    model = AutoModelForObjectDetection.from_config(config, **model_kwargs)

    print(f"  Loading weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    return model, processor


def create_model_card(
    model_name: str, map_score: str, hf_base: str, repo_id: str
) -> str:
    """Generate a README model card."""
    class_table = "\n".join(f"| {i} | {name} |" for i, name in enumerate(CLASS_NAMES))

    return f"""---
license: apache-2.0
tags:
- object-detection
- fruit-detection
- transformers
- {model_name}
datasets:
- custom
metrics:
- map
pipeline_tag: object-detection
---

# Fruit Detector - {model_name}

This model is a fine-tuned version of [{hf_base}](https://huggingface.co/{hf_base}) for fruit and vegetable detection.

## Model Details

- **Base Model:** {hf_base}
- **Architecture:** {model_name}
- **Task:** Object Detection
- **mAP@50 Score:** {map_score}
- **Input Size:** 640x640

## Classes

The model detects the following 12 fruit/vegetable classes:

| ID | Class |
|----|-------|
{class_table}

## Usage

```python
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image
import torch

# Load model and processor
processor = AutoImageProcessor.from_pretrained("{repo_id}")
model = AutoModelForObjectDetection.from_pretrained("{repo_id}")

# Load and process image
image = Image.open("fruit_image.jpg")
inputs = processor(images=image, return_tensors="pt")

# Run inference
with torch.no_grad():
    outputs = model(**inputs)

# Post-process results
target_sizes = torch.tensor([[image.height, image.width]])
results = processor.post_process_object_detection(
    outputs,
    threshold=0.5,
    target_sizes=target_sizes
)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = box.tolist()
    print(f"Detected {{model.config.id2label[label.item()]}} with confidence {{score:.2f}} at {{box}}")
```

## Training

This model was trained on a custom fruit detection dataset.

**Training Repository:** [transformers-for-fruit-object-detection-internship](https://github.com/mohamedkhayat/transformers-for-fruit-object-detection-internship)

## License

Apache 2.0
"""


def upload_to_hub(
    checkpoint_path: str,
    repo_name: str,
    hf_token: str,
    organization: str = None,
    private: bool = False,
):
    """Upload model to HuggingFace Hub."""
    model_name = extract_model_name(checkpoint_path)
    map_score = extract_map_score(checkpoint_path)
    hf_base = SUPPORTED_MODELS[model_name]

    print(f"📦 Model architecture: {model_name}")
    print(f"📊 mAP Score: {map_score}")
    print(f"🔗 Base model: {hf_base}")

    print("\n⏳ Loading model and processor...")
    model, processor = load_and_prepare_model(checkpoint_path, model_name)

    api = HfApi(token=hf_token)
    if organization:
        repo_id = f"{organization}/{repo_name}"
    else:
        user_info = api.whoami()
        repo_id = f"{user_info['name']}/{repo_name}"

    print(f"\n🚀 Uploading to: {repo_id}")

    try:
        create_repo(repo_id, token=hf_token, private=private, exist_ok=True)
        print(f"  Repository created/verified: {repo_id}")
    except Exception as e:
        print(f"  Note: {e}")

    save_dir = f"/tmp/hf_upload_{repo_name}"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    print("\n💾 Saving model to temporary directory...")

    model.save_pretrained(save_dir, safe_serialization=True)
    print("  ✓ Model saved (safetensors format)")

    processor.save_pretrained(save_dir)
    print("  ✓ Processor saved")

    model_card = create_model_card(model_name, map_score, hf_base, repo_id)
    with open(os.path.join(save_dir, "README.md"), "w") as f:
        f.write(model_card)
    print("  ✓ Model card created")

    print("\n📁 Files to upload:")
    for f in os.listdir(save_dir):
        size = os.path.getsize(os.path.join(save_dir, f))
        size_str = (
            f"{size / 1024 / 1024:.1f} MB"
            if size > 1024 * 1024
            else f"{size / 1024:.1f} KB"
        )
        print(f"  - {f} ({size_str})")

    print("\n⬆️  Uploading to HuggingFace Hub...")
    api.upload_folder(
        folder_path=save_dir,
        repo_id=repo_id,
        repo_type="model",
    )

    shutil.rmtree(save_dir)

    print("\n✅ Successfully uploaded!")
    print(f"🔗 View at: https://huggingface.co/{repo_id}")
    print("\n📝 Usage in Python:")
    print("   from transformers import AutoImageProcessor, AutoModelForObjectDetection")
    print(f'   processor = AutoImageProcessor.from_pretrained("{repo_id}")')
    print(f'   model = AutoModelForObjectDetection.from_pretrained("{repo_id}")')

    return repo_id


def list_available_checkpoints(checkpoint_dir: str = "checkpoints/save"):
    """List all available checkpoints."""
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return

    checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")])

    if not checkpoints:
        print(f"❌ No .pth files found in: {checkpoint_dir}")
        return

    print(f"\n📦 Available checkpoints in {checkpoint_dir}:\n")
    for ckpt in checkpoints:
        try:
            model_name = extract_model_name(ckpt)
            map_score = extract_map_score(ckpt)
            print(f"  • {ckpt}")
            print(f"    Model: {model_name}, mAP: {map_score}")
        except ValueError:
            print(f"  • {ckpt} (could not parse)")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Upload fruit detection model to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available checkpoints
  python upload_to_hf.py --list

  # Upload a model
  python upload_to_hf.py \\
      --checkpoint checkpoints/save/model-rtdetrv2_50_lr-0.0001_0902_2210_0.6506.pth \\
      --repo-name fruit-detector-rtdetrv2-50

  # Upload to an organization
  python upload_to_hf.py \\
      --checkpoint checkpoints/save/model-dfine_xlarge_coco_lr-0.0001_0906_0933_0.6631.pth \\
      --repo-name fruit-detector-dfine-xlarge \\
      --organization your-org-name

  # Make it private
  python upload_to_hf.py \\
      --checkpoint checkpoints/save/model-rtdetrv1_101_lr-0.0001_0904_1314_0.6775.pth \\
      --repo-name fruit-detector-rtdetrv1-101 \\
      --private
        """,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to the .pth checkpoint file",
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        help="Name for the HuggingFace repository",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace API token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--organization",
        type=str,
        default=None,
        help="Organization to upload to (optional, defaults to your username)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available checkpoints and exit",
    )

    args = parser.parse_args()

    if args.list:
        list_available_checkpoints()
        return 0

    if not args.checkpoint:
        print("❌ Error: --checkpoint is required")
        print("   Use --list to see available checkpoints")
        return 1

    if not args.repo_name:
        print("❌ Error: --repo-name is required")
        return 1

    if not args.hf_token:
        print("❌ Error: HuggingFace token required")
        print("   Use --hf-token or set HF_TOKEN environment variable")
        print("   Get your token from: https://huggingface.co/settings/tokens")
        return 1

    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint not found: {args.checkpoint}")
        return 1

    try:
        upload_to_hub(
            checkpoint_path=args.checkpoint,
            repo_name=args.repo_name,
            hf_token=args.hf_token,
            organization=args.organization,
            private=args.private,
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
