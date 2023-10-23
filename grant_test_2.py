import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchgeo.datasets import VHR10
from PIL import Image

# Define the resize transform
resize_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((1024, 1024)),
    transforms.ToTensor()
])

# Custom collate function
def custom_collate(batch):
    images = [item["image"] for item in batch]
    labels = [item["labels"] for item in batch]

    resized_images = [resize_transform(img) for img in images]
    resized_images = torch.stack(resized_images)

    # Since labels can have different lengths, we keep them as a list instead of stacking
    return {"image": resized_images, "labels": labels}

# Initialize the dataset
dataset = VHR10(root="./raw_data", download=True, checksum=True)

# Initialize the dataloader with the custom collate function
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4, collate_fn=custom_collate)

# Training loop
for batch in dataloader:
    image = batch["image"]
    labels = batch["labels"]

    # Continue with your model training or prediction logic
