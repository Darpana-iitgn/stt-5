import os
import random
import shutil
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
from transformers import ResNetForImageClassification, ResNetConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from augly.image import functional as augly
from PIL import Image

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Define dataset paths
original_dataset_path = "dataset-dog-cat"  # Ensure this contains 'cats' and 'dogs' folders
train_path = "dataset/train"
test_path = "dataset/test_split"
augmented_path = "dataset/augmented_train"
final_train_path = "dataset/final_train"

# Create necessary directories
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)
os.makedirs(augmented_path, exist_ok=True)
os.makedirs(final_train_path, exist_ok=True)
for category in ['cats', 'dogs']:
    os.makedirs(os.path.join(train_path, category), exist_ok=True)
    os.makedirs(os.path.join(test_path, category), exist_ok=True)
    os.makedirs(os.path.join(augmented_path, category), exist_ok=True)
    os.makedirs(os.path.join(final_train_path, category), exist_ok=True)

# Splitting dataset into train (80%) and test (20%)
def split_dataset():
    for category in ['cats', 'dogs']:
        category_path = os.path.join(original_dataset_path, category)
        files = os.listdir(category_path)
        random.shuffle(files)
        train_files, test_files = files[:400], files[400:500]  # 500 total, 400 train, 100 test
        
        for file in train_files:
            shutil.copy(os.path.join(category_path, file), os.path.join(train_path, category, file))
        for file in test_files:
            shutil.copy(os.path.join(category_path, file), os.path.join(test_path, category, file))

split_dataset()

# Define augmentation function using Augly
def augment_image(image):
    augmentations = [
        augly.rotate,
        augly.blur,
        augly.crop,
        augly.hflip,
        augly.vflip,
        augly.saturation,
        augly.brightness,
        augly.sharpen,
        augly.noise,
        augly.color_jitter
    ]
    augmented_images = []
    for _ in range(2):  # Augment twice per image
        random.shuffle(augmentations)
        aug_image = image
        for aug in augmentations[:3]:  # Apply 3 random augmentations
            aug_image = aug(aug_image)
        augmented_images.append(aug_image)
    return augmented_images

# Perform data augmentation on train set
def augment_dataset():
    for category in ['cats', 'dogs']:
        images = os.listdir(os.path.join(train_path, category))
        for image_file in images:
            image = Image.open(os.path.join(train_path, category, image_file)).convert("RGB")
            augmented_images = augment_image(image)
            for i, aug_image in enumerate(augmented_images):
                aug_image.save(os.path.join(augmented_path, category, f"aug_{i}_{image_file}"))
    
augment_dataset()

# Merge original train set with augmented images
def merge_datasets():
    for category in ['cats', 'dogs']:
        for file in os.listdir(os.path.join(train_path, category)):
            shutil.copy(os.path.join(train_path, category, file), os.path.join(final_train_path, category, file))
        for file in os.listdir(os.path.join(augmented_path, category)):
            shutil.copy(os.path.join(augmented_path, category, file), os.path.join(final_train_path, category, file))

merge_datasets()

# Show dataset statistics
def plot_dataset_statistics():
    categories = ['cats', 'dogs']
    counts = {
        'Original Train': [len(os.listdir(os.path.join(train_path, c))) for c in categories],
        'Test': [len(os.listdir(os.path.join(test_path, c))) for c in categories],
        'Augmented': [len(os.listdir(os.path.join(augmented_path, c))) for c in categories],
        'Final Train': [len(os.listdir(os.path.join(final_train_path, c))) for c in categories]
    }
    
    df = []
    for key, values in counts.items():
        for cat, count in zip(categories, values):
            df.append([key, cat, count])
    
    import pandas as pd
    df = pd.DataFrame(df, columns=['Dataset', 'Category', 'Count'])
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Dataset', y='Count', hue='Category', data=df)
    plt.title("Dataset Statistics")
    plt.show()
    
plot_dataset_statistics()

# Data Loading with PyTorch
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(root=final_train_path, transform=data_transforms)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

def train_model(train_loader, test_loader, use_augmented=False):
    model = ResNetForImageClassification(ResNetConfig(num_labels=2))
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    num_epochs = 5
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
    
    # Evaluate model
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images).logits
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_true.extend(labels.numpy())
            y_pred.extend(preds)
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

# Train model with final dataset
train_model(dataloader, dataloader, use_augmented=True)
