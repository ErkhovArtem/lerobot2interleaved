# Lerobot2Interleaved Converter

A set of scripts for working with LeRobot datasets: conversion to interleave format and visualization of resulted image instruction.

## 🔧 Scripts

## 1. Convert to Interleave Format

Purpose: Converts regular LeRobot datasets to interleave format (text + image instructions).

Original dataset structure is preserved; new fields are added to episodes.jsonl

### Features:

- Adds visual object references to instructions

- Maintains compatibility with original LeRobot format

- Automatically detects objects using OWLv2

- Preserves all original dataset structure

### Usage:

```
python convert_lerobot_interleave.py \
  --repo-id "${HF_USER}/<dataset>" \
  --output ./output_interleave \
  --objects "cup" "mug" "bottle" "spoon" \
  --device cuda:0
```

### Parameters:

--repo-id: Dataset ID on Hugging Face Hub or local path

--output: Path for converted dataset

--objects: List of objects to detect (space-separated)

--device: Computation device (cuda:0, cpu)

--root: Local dataset root directory (optional)

### Output format:

```
{
    "interleaved_instruction": {
        "interleaved_text": "pick up the <image> and place on the <image>",
        "original_text": "pick up the red cup and place on the table",
        "image_instruction": [[RGB arrays...]],
        "image_mask": [True, True],
        "object_order": ["red cup", "table"],
        "detected_objects": ["red cup", "table"]
    }
}
```

## 2. Episode Visualization

Purpose: Visualizes dataset episodes with interleaved instructions and object annotations.

### Features:

- Displays instruction images with target objects

- Shows original and interleaved text instructions

- Annotates images with object labels and metadata

- Supports both simple PIL and advanced matplotlib visualization

- Saves visualization results

### Usage:

```
python visualize_episode.py \
  --repo-id "${HF_USER}/<dataset in interleaved format>" \
  --episode_idx 5
```

### Parameters:

--repo-id: Dataset to visualize

--episode_idx: Episode index to display (default: 0)

--root: Local dataset root (optional)

### Output: 

Displays image instruction and true label.

## 📋 Requirements

Install lerobot from official github https://github.com/huggingface/lerobot.

To use OWLv2 detector:

```
pip install transformers scipy
```