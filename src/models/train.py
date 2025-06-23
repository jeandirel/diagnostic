
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.utils.config import TRAIN_DIR, TRAIN_ANNOTATIONS, VALID_DIR, VALID_ANNOTATIONS
from src.utils.data_loader import get_dataloader
from src.models.unet import UNet

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load data
train_loader = get_dataloader(TRAIN_DIR, TRAIN_ANNOTATIONS, batch_size=4, shuffle=True)
val_loader = get_dataloader(VALID_DIR, VALID_ANNOTATIONS, batch_size=4, shuffle=False)

# Initialize model for 14 classes (0=background + 13 target classes)
model = UNet(in_channels=3, out_channels=14).to(device)

# Loss and optimizer
class_weights = torch.tensor(
    [1.0] + [5.0] * 13,  # background weight = 1, vertebrae and heart = 5
    dtype=torch.float32
).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=5e-5)  # Lower learning rate

# Early stopping params
best_val_loss = float('inf')
patience = 5
counter = 0

# Training loop
epochs = 50
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)  # shape: [B, H, W]

        outputs = model(images)   # shape: [B, 14, H, W]
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}")

    # Validation loss
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_images, val_masks in val_loader:
            val_images = val_images.to(device)
            val_masks = val_masks.to(device)
            val_outputs = model(val_images)
            val_loss += criterion(val_outputs, val_masks).item() * val_images.size(0)

    val_loss /= len(val_loader.dataset)
    print(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}")

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), "unet_model.pth")
        print("Model improved and saved.")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break

print("Training complete.")