
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from unet_plus_plus import UNetPlusPlus
from datasets import DentexDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# Configuration
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 1e-4
IMAGE_SIZE = 256
NUM_CLASSES = 4  # Impacted, Caries, Periapical Lesion, Deep Caries
PATIENCE = 5  # Number of epochs to wait before stopping if no improvement

# Directories
TRAIN_IMAGE_DIR = "DENTEX/training_data/quadrant-enumeration-disease/xrays"
TRAIN_MASK_DIR = "DENTEX/training_data/quadrant-enumeration-disease/train_quadrant_enumeration_disease.json"
VALIDATION_IMAGE_DIR = "DENTEX/validation_data/quadrant_enumeration_disease/xrays"
VALIDATION_MASK_DIR = "DENTEX/validation_data/validation_triple.json"

# Data Augmentation
transform = A.Compose([
    A.HorizontalFlip(),
    A.RandomBrightnessContrast(),
    A.Rotate(limit=30),
    A.Resize(height=256, width=256),
    ToTensorV2()
])


# Loss Function (Dice Loss + CrossEntropy)
def dice_loss(pred, target, smooth=1e-6):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1 - (2. * intersection + smooth) / (union + smooth)


if __name__ == '__main__':
    # Dataset and DataLoader
    train_dataset = DentexDataset(
        image_dir=TRAIN_IMAGE_DIR,
        label_file=TRAIN_MASK_DIR,
        augmentations=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    val_dataset = DentexDataset(
        image_dir=VALIDATION_IMAGE_DIR,
        label_file=VALIDATION_MASK_DIR,
        augmentations=transform
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Model Initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetPlusPlus(in_channels=3, out_channels=NUM_CLASSES).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')  # Initialize the best validation loss as infinity
    epochs_no_improve = 0  # Counter for epochs without improvement
    early_stop = False  # Flag to stop training early

    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False)

        for batch_idx, (images, masks) in enumerate(loop):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = dice_loss(outputs, masks)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            loop.set_postfix(loss=loss.item())  # Show current batch loss

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}] - Avg Loss: {avg_loss:.4f}")

        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"checkpoint_epoch_{epoch + 1}.pth")

        # Validation Loss Check for Early Stopping
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = dice_loss(outputs, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # Check if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            epochs_no_improve += 1
            print(f"Early stopping counter: {epochs_no_improve}/{PATIENCE}")

        # Early stopping condition
        if epochs_no_improve >= PATIENCE:
            print("Early stopping triggered!")
            early_stop = True
            break

        # Visualization after each epoch
        if epoch % 5 == 0:  # Plot results after every 5 epochs, for example
            with torch.no_grad():
                image, mask = next(iter(train_loader))
                output = model(image.to(device))
                output = torch.argmax(output, dim=1)  # For multi-class segmentation

                # Plot the result
                plt.subplot(1, 3, 1)
                plt.imshow(image[0].cpu().numpy().transpose(1, 2, 0))
                plt.title("Input Image")
                plt.subplot(1, 3, 2)
                plt.imshow(mask[0].cpu().numpy(), cmap='gray')
                plt.title("Ground Truth")
                plt.subplot(1, 3, 3)
                plt.imshow(output[0].cpu().numpy(), cmap='gray')
                plt.title("Prediction")
                plt.show()

    if not early_stop:
        torch.save(model.state_dict(), "final_model.pth")
        print("Training completed and model saved.")
    else:
        print("Training stopped early due to no improvement in validation loss.")
