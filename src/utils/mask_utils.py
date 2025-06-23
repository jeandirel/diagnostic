import numpy as np
from pycocotools import mask as maskUtils

# Define your consistent class-to-index mapping
CLASS_NAME_TO_ID = {
    "heart": 1,
    "vertebrae_thoracic_1": 2,
    "vertebrae_thoracic_2": 3,
    "vertebrae_thoracic_3": 4,
    "vertebrae_thoracic_4": 5,
    "vertebrae_thoracic_5": 6,
    "vertebrae_thoracic_6": 7,
    "vertebrae_thoracic_7": 8,
    "vertebrae_thoracic_8": 9,
    "vertebrae_thoracic_9": 10,
    "vertebrae_thoracic_10": 11,
    "vertebrae_thoracic_11": 12,
    "vertebrae_thoracic_12": 13
}

def create_multiclass_mask(annotations, image_size):
    """
    Convert a list of annotations (for one image) to a single-channel multi-class mask.

    Args:
        annotations (list): List of annotations (COCO-style) for one image.
        image_size (tuple): (height, width) of the image.

    Returns:
        np.ndarray: Integer mask of shape (H, W) with values [0, num_classes]
    """
    height, width = image_size
    mask = np.zeros((height, width), dtype=np.uint8)  # 0 = background

    for ann in annotations:
        segmentation = ann.get("segmentation", [])
        class_name = ann.get("category_name")
        class_id = CLASS_NAME_TO_ID.get(class_name)

        if class_id is None or not segmentation:
            continue

        rles = maskUtils.frPyObjects(segmentation, height, width)
        rle = maskUtils.merge(rles)
        ann_mask = maskUtils.decode(rle)

        # Set pixels where ann_mask == 1 to class_id
        mask[ann_mask == 1] = class_id

    return mask
