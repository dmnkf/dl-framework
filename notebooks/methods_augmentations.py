# %% [code]
import sys
from pathlib import Path
import random
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import functional as F

# add project root to sys.path so that imports work correctly
project_root = Path().absolute().parent
sys.path.append(str(project_root))

from src.data.unified import UnifiedDataset
from src.data.dataset import DatasetModality

# For reproducibility
random.seed(42)
torch.manual_seed(42)

# %% [code]
# Load the dataset record.
data_root = Path(project_root) / "data"
acdc_data = UnifiedDataset(data_root, modality=DatasetModality.CMR, dataset_key="acdc")
record = acdc_data["patient001"]

# Check the shape of the input tensor (expected shape: [3, H, W])
print("Input shape:", record.preprocessed_record.inputs.shape)


# %% [code]
# Helper function to convert a tensor image to a PIL image.
def to_pil(img):
    # If already a PIL image, return directly.
    if hasattr(img, "convert"):
        return img
    # Assume input is a torch tensor in [C, H, W] with values in [0, 1] or [0,255]
    return F.to_pil_image(img)


# %% [code]
# Define transformation pipelines.
# Assume an image size based on the input tensor (e.g., 210).
img_size = record.preprocessed_record.inputs.shape[-1]

# Default pipeline: simply resize and convert to float tensor.
default_pipeline = transforms.Compose(
    [transforms.Resize((img_size, img_size)), transforms.Lambda(lambda x: x.float())]
)

# Augmentation pipeline: horizontal flip, rotation, color jitter, random resized crop, and conversion.
rotation_degrees = 45
brightness = 0.5
contrast = 0.5
saturation = 0.25
random_crop_scale = (0.6, 1.0)

augment_pipeline = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(rotation_degrees),
        transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation
        ),
        transforms.RandomResizedCrop(size=img_size, scale=random_crop_scale),
        transforms.Lambda(lambda x: x.float()),
    ]
)


# %% [code]
# Define a helper function that applies a transformation pipeline step by step per channel.
def apply_pipeline_steps_per_slice(image, pipeline):
    """
    Applies each transform in a Compose pipeline step by step and returns a list where each element is
    a tuple (step_name, list of transformed images for each channel).
    """
    # Convert each channel of the input tensor to a PIL image.
    channels = [to_pil(image[i]) for i in range(image.shape[0])]
    steps = [("Original", channels)]

    # Process the transformations sequentially.
    current_channels = channels.copy()
    for transform in pipeline.transforms:
        next_channels = []
        for ch in current_channels:
            try:
                transformed = transform(ch)
            except Exception as e:
                print(f"Error applying {transform}: {e}")
                transformed = ch
            # Ensure the output is in PIL format.
            if not hasattr(transformed, "convert"):
                transformed = to_pil(transformed)
            next_channels.append(transformed)
        steps.append((transform.__class__.__name__, next_channels))
        current_channels = next_channels
    return steps


# %% [code]
# jsut plot signle imqge raw
plt.imshow(to_pil(record.preprocessed_record.inputs[0]), cmap="gray")
plt.axis("off")
plt.show()


# %% [code]
# Visualize the original input image (all 3 slices shown as separate subplots).
input_image = record.preprocessed_record.inputs  # shape: [3, H, W]
fig, axs = plt.subplots(1, input_image.shape[0], figsize=(4 * input_image.shape[0], 4))
for ax, idx in zip(axs, range(input_image.shape[0])):
    ax.imshow(to_pil(input_image[idx]), cmap="gray")
    ax.set_title(f"Slice {idx+1}")
    ax.axis("off")
plt.suptitle("Original Cine CMR Slices", fontsize=16)
plt.tight_layout()
plt.show()

# %% [code]
# Visualize the step-by-step effects of the augmentation pipeline on each slice.
steps_augment = apply_pipeline_steps_per_slice(input_image, augment_pipeline)

for step_name, channel_imgs in steps_augment:
    n_channels = len(channel_imgs)
    fig, axs = plt.subplots(1, n_channels, figsize=(4 * n_channels, 4))
    for ax, img in zip(axs, channel_imgs):
        ax.imshow(img, cmap="gray")
        ax.axis("off")
    fig.suptitle(f"Step: {step_name}", fontsize=16)
    plt.tight_layout()
    plt.show()

# %% [code]
# Optionally, visualize the default pipeline steps in a similar fashion.
steps_default = apply_pipeline_steps_per_slice(input_image, default_pipeline)

for step_name, channel_imgs in steps_default:
    n_channels = len(channel_imgs)
    fig, axs = plt.subplots(1, n_channels, figsize=(4 * n_channels, 4))
    for ax, img in zip(axs, channel_imgs):
        ax.imshow(img, cmap="gray")
        ax.axis("off")
    fig.suptitle(f"Default Pipeline Step: {step_name}", fontsize=16)
    plt.tight_layout()
    plt.show()
