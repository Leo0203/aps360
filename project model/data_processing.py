import os
import random
import time
import numpy as np
from PIL import Image
from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torchvision import transforms

########################################only run for raw data########################################


# Resize transform
resize_transform = transforms.Resize((224, 224))

# Change these to wherever your raw images live (and where you want cleaned copies to go)
# For train dataset
# input_dir = r'/Users/mashilin/Desktop/study/coding/APS360/project/raw_dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train'
# output_dir = r'/Users/mashilin/Desktop/study/coding/APS360/project/clean_dataset/train'

# For validation dataset
# input_dir = r'/Users/mashilin/Desktop/study/coding/APS360/project/raw_dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid'
# output_dir = r'/Users/mashilin/Desktop/study/coding/APS360/project/clean_dataset/val'

# For test dataset
input_dir = r'/Users/mashilin/Desktop/study/coding/APS360/project/raw_dataset/test'
output_dir = r'/Users/mashilin/Desktop/study/coding/APS360/project/clean_dataset/test'

# Walk each class-folder, resize and save
for class_folder in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_folder)
    if not os.path.isdir(class_path):
        continue

    output_class_path = os.path.join(output_dir, class_folder)
    os.makedirs(output_class_path, exist_ok=True)

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        try:
            with Image.open(img_path).convert('RGB') as img:
                img_resized = resize_transform(img)
                save_path = os.path.join(output_class_path, img_name)
                img_resized.save(save_path)
        except Exception as e:
            # you could print(e, img_path) to debug bad files
            continue

# Map gesture folder names to class indices (e.g., {'D': 0, 'A': 1, ...})
gesture_folders = sorted(os.listdir(output_dir))  # Sorted ensures consistent mapping
gesture_to_label = {g: i for i, g in enumerate(gesture_folders)}
print("Label mapping:", gesture_to_label)

for folder in os.listdir(output_dir):
    folder_path = os.path.join(output_dir, folder)
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        try:
            img = Image.open(img_path)
            if img.mode != 'RGB':
                os.remove(img_path)
        except:
            os.remove(img_path)  # Remove corrupted