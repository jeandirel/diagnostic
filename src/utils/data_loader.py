import os
import json
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from src.utils.config import TRAIN_ANNOTATIONS, TRAIN_DIR, VALID_ANNOTATIONS, VALID_DIR
from src.utils.mask_utils import create_multiclass_mask

class PetRadiologyDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        with open(annotation_file, "r") as f:
            coco_data = json.load(f)

        self.image_info = coco_data["images"]
        self.annotations = coco_data["annotations"]
        self.categories = coco_data["categories"]
        # Build category_id to name mapping
        self.category_id_to_name = {cat["id"]: cat["name"] for cat in self.categories}

        # Group annotations by image_id
        self.ann_by_image = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            ann["category_name"] = self.category_id_to_name[ann["category_id"]]  # Inject name
            if img_id not in self.ann_by_image:
                self.ann_by_image[img_id] = []
            self.ann_by_image[img_id].append(ann)

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        image_info = self.image_info[idx]
        image_id = image_info["id"]
        filename = image_info["file_name"]

        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        annotations = self.ann_by_image.get(image_id, [])
        multiclass_mask = create_multiclass_mask(annotations, (height, width))
        mask_pil = Image.fromarray(multiclass_mask)

        if self.transform:
            image = self.transform(image)
            mask = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST)(mask_pil)
            mask = torch.from_numpy(np.array(mask)).long()  # Convert to tensor with int labels
        else:
            mask = torch.from_numpy(multiclass_mask).long()

        return image, mask

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def get_dataloader(image_dir, annotation_file, batch_size=8, shuffle=True):
    dataset = PetRadiologyDataset(image_dir, annotation_file, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchvision.transforms.functional import to_pil_image

    dataset = PetRadiologyDataset(TRAIN_DIR, TRAIN_ANNOTATIONS, transform=transform)
    image, mask = dataset[0]

    plt.imshow(to_pil_image(image))
    plt.title("Image")
    plt.axis("off")
    plt.show()

    plt.imshow(mask, cmap="tab20")
    plt.title("Multi-class Mask")
    plt.axis("off")
    plt.show()