"""
COCO Segmentation → Semantic Mask Conversion
==============================================
Converts Roboflow COCO segmentation annotations (RLE format) into
single-channel PNG semantic segmentation masks for CondSeg training.

COCO category IDs (from Roboflow):
    0: eye-dELd (eyelid / eye region boundary — we skip this, it's implicit)
    1: iris
    2: pupil
    3: sclera

CondSeg target mask class IDs:
    0: Background
    1: Sclera
    2: Iris
    3: Pupil

Mapping:  COCO cat 3 (sclera) → class 1
          COCO cat 1 (iris)   → class 2
          COCO cat 2 (pupil)  → class 3
          COCO cat 0 (eye-dELd) → skipped (background is implicit)

The script also saves random colorized samples (image + overlay) to a
test_vis/ directory so you can visually verify the conversion.

Usage:
    python coco_to_masks.py
"""

import json
import os
import random
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
import shutil


# ============================================================================
#  CONFIGURATION
# ============================================================================

# Root of the COCO dataset (Roboflow export)
COCO_ROOT = r"c:\Users\aliya\Downloads\cocoseg"

# Output root for converted data (images + masks side by side)
OUTPUT_ROOT = r"c:\Users\aliya\workspace\TonoCAM\CondSeg\data"

# How many random samples to save with colorized overlay for visual check
NUM_VIS_SAMPLES = 10

# Mapping: COCO category_id → CondSeg class ID
# We skip 'eye-dELd' (cat 0) as it defines the eye region boundary,
# which is implicit (everything non-background).
COCO_TO_CONDSEG = {
    3: 1,   # sclera  → class 1
    1: 2,   # iris    → class 2
    2: 3,   # pupil   → class 3
    # 0: skip (eye-dELd)
}

# Priority: higher class ID overwrites lower on overlap (pupil > iris > sclera)
# This naturally happens when we paint in order: sclera first, iris, then pupil.
PAINT_ORDER = [3, 1, 2]  # COCO cat IDs in painting order (sclera, iris, pupil)

# Colors for visualization overlay (RGB)
VIS_COLORS = {
    0: (0,   0,   0),     # Background — black
    1: (255, 255, 255),   # Sclera     — white
    2: (0,   180, 80),    # Iris       — green
    3: (255, 50,  50),    # Pupil      — red
}


# ============================================================================
#  CONVERSION FUNCTIONS
# ============================================================================

def decode_rle_annotation(ann: dict, img_h: int, img_w: int) -> np.ndarray:
    """
    Decode a COCO annotation's segmentation field into a binary mask.

    Handles both RLE (dict with 'counts', 'size') and polygon formats.

    Returns:
        binary_mask: (H, W) numpy array, uint8, values {0, 1}
    """
    seg = ann["segmentation"]

    if isinstance(seg, dict):
        # RLE format (from Roboflow)
        rle = seg
        # Ensure counts is in the right format
        if isinstance(rle["counts"], list):
            rle = mask_utils.frPyObjects(rle, img_h, img_w)
        binary_mask = mask_utils.decode(rle)  # (H, W), uint8
    elif isinstance(seg, list):
        # Polygon format → convert to RLE first
        rles = mask_utils.frPyObjects(seg, img_h, img_w)
        rle = mask_utils.merge(rles)
        binary_mask = mask_utils.decode(rle)
    else:
        raise ValueError(f"Unknown segmentation format: {type(seg)}")

    return binary_mask.astype(np.uint8)


def build_semantic_mask(
    annotations: list,
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """
    Build a single-channel semantic segmentation mask from COCO annotations.

    Paints classes in order (sclera → iris → pupil) so higher-priority
    classes overwrite lower ones on overlap.

    Returns:
        mask: (H, W) numpy array, uint8, values in {0, 1, 2, 3}
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)  # all background

    for coco_cat_id in PAINT_ORDER:
        condseg_class = COCO_TO_CONDSEG.get(coco_cat_id)
        if condseg_class is None:
            continue

        # Find all annotations for this category
        cat_anns = [a for a in annotations if a["category_id"] == coco_cat_id]

        for ann in cat_anns:
            binary = decode_rle_annotation(ann, img_h, img_w)
            mask[binary > 0] = condseg_class

    return mask


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Convert a single-channel class mask to an RGB colorized image.

    Returns:
        colored: (H, W, 3) numpy array, uint8
    """
    H, W = mask.shape
    colored = np.zeros((H, W, 3), dtype=np.uint8)
    for class_id, color in VIS_COLORS.items():
        colored[mask == class_id] = color
    return colored


def create_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Create a blended overlay of image + colorized mask.

    Returns:
        overlay: (H, W, 3) numpy array, uint8
    """
    colored = colorize_mask(mask)
    # Only blend where mask is non-zero (non-background)
    overlay = image.copy()
    fg = mask > 0
    overlay[fg] = (alpha * colored[fg] + (1 - alpha) * image[fg]).astype(np.uint8)
    return overlay


# ============================================================================
#  MAIN CONVERSION
# ============================================================================

def convert_split(split: str, vis_indices: set = None):
    """
    Convert a single data split (train/valid/test).

    Args:
        split:       One of 'train', 'valid', 'test'
        vis_indices: Set of image indices to save visual samples for
    """
    src_dir = os.path.join(COCO_ROOT, split)
    json_path = os.path.join(src_dir, "_annotations.coco.json")

    if not os.path.exists(json_path):
        print(f"  ⚠ No annotations found for '{split}', skipping.")
        return 0

    # Load COCO JSON
    with open(json_path, "r") as f:
        coco = json.load(f)

    # Build lookup: image_id → list of annotations
    ann_by_image = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        ann_by_image.setdefault(img_id, []).append(ann)

    # Output directories
    out_img_dir  = os.path.join(OUTPUT_ROOT, split, "images")
    out_mask_dir = os.path.join(OUTPUT_ROOT, split, "masks")
    out_vis_dir  = os.path.join(OUTPUT_ROOT, "test_vis")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)
    os.makedirs(out_vis_dir, exist_ok=True)

    if vis_indices is None:
        vis_indices = set()

    count = 0
    for idx, img_info in enumerate(coco["images"]):
        img_id = img_info["id"]
        img_fname = img_info["file_name"]
        img_h = img_info["height"]
        img_w = img_info["width"]

        # Source image path
        src_img_path = os.path.join(src_dir, img_fname)
        if not os.path.exists(src_img_path):
            print(f"    ⚠ Image not found: {img_fname}")
            continue

        # Get annotations for this image
        anns = ann_by_image.get(img_id, [])

        # Build semantic mask
        mask = build_semantic_mask(anns, img_h, img_w)

        # Generate output filenames (use a clean basename without extension)
        base_name = os.path.splitext(img_fname)[0]

        # Copy image to output
        dst_img_path = os.path.join(out_img_dir, img_fname)
        shutil.copy2(src_img_path, dst_img_path)

        # Save mask as PNG (single-channel, values 0-3)
        mask_img = Image.fromarray(mask, mode="L")
        dst_mask_path = os.path.join(out_mask_dir, f"{base_name}.png")
        mask_img.save(dst_mask_path)

        # Save visualization if this is a selected sample
        if idx in vis_indices:
            image_np = np.array(Image.open(src_img_path).convert("RGB"))
            overlay = create_overlay(image_np, mask, alpha=0.5)
            colored = colorize_mask(mask)

            # Save: original | colorized mask | overlay (side by side)
            h, w = image_np.shape[:2]
            canvas = np.zeros((h, w * 3, 3), dtype=np.uint8)
            canvas[:, :w] = image_np
            canvas[:, w:2*w] = colored
            canvas[:, 2*w:] = overlay

            vis_path = os.path.join(out_vis_dir, f"{split}_{base_name}_vis.jpg")
            Image.fromarray(canvas).save(vis_path, quality=95)

        count += 1

    return count


def main():
    print("=" * 60)
    print("COCO Segmentation → Semantic Mask Conversion")
    print("=" * 60)
    print(f"  Source : {COCO_ROOT}")
    print(f"  Output : {OUTPUT_ROOT}")
    print()

    # Clean output if exists
    if os.path.exists(OUTPUT_ROOT):
        print("  Cleaning previous output...")
        shutil.rmtree(OUTPUT_ROOT)

    total = 0
    for split in ["train", "valid", "test"]:
        print(f"\n  Processing '{split}' split...")

        # Determine how many images in this split for vis sample selection
        json_path = os.path.join(COCO_ROOT, split, "_annotations.coco.json")
        if not os.path.exists(json_path):
            continue

        with open(json_path, "r") as f:
            n_images = len(json.load(f)["images"])

        # Pick random indices for visualization
        n_vis = min(NUM_VIS_SAMPLES, n_images)
        vis_indices = set(random.sample(range(n_images), n_vis))

        count = convert_split(split, vis_indices)
        total += count
        print(f"    ✓ {count} images converted, {n_vis} visual samples saved")

    print(f"\n{'=' * 60}")
    print(f"  DONE: {total} total images converted")
    print(f"  Masks saved to : {OUTPUT_ROOT}/<split>/masks/")
    print(f"  Visual samples : {OUTPUT_ROOT}/test_vis/")
    print(f"{'=' * 60}")

    # Print class distribution from a random mask for sanity check
    print("\n  Sanity check — class pixel counts from a random train mask:")
    mask_dir = os.path.join(OUTPUT_ROOT, "train", "masks")
    if os.path.exists(mask_dir):
        masks = os.listdir(mask_dir)
        if masks:
            sample = np.array(Image.open(os.path.join(mask_dir, random.choice(masks))))
            for cls_id, cls_name in [(0, "Background"), (1, "Sclera"), (2, "Iris"), (3, "Pupil")]:
                pct = (sample == cls_id).sum() / sample.size * 100
                print(f"    Class {cls_id} ({cls_name:>10s}): {pct:6.2f}%")


if __name__ == "__main__":
    main()
