import os
import numpy as np
import seaborn as sns
from PIL import Image
from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset, WeightedRandomSampler, Dataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import re

'''
##################################use for entire dataset##################################
data_loc = 'C:\\Users\\16482\\Desktop\\aps360\\cleandata\\cleandata'


def get_data_loader(data_loc, batch_size,
                    val_pct=0.1, test_pct=0.1,
                    seed=1000, num_workers=0):

    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=30),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.9, contrast=0.9, saturation=0.9, hue=0.1),
        transforms.RandomAffine(degrees=0, scale=(0.8, 1.0)),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 0.2))], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load full dataset (only to get targets)
    full_ds_plain = datasets.ImageFolder(root=data_loc)
    targets = np.array(full_ds_plain.targets)

    # First split: train vs temp (val + test)
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        np.arange(len(targets)), targets,
        test_size=val_pct + test_pct,
        stratify=targets,
        random_state=seed
    )

    # Second split: val vs test
    val_pct_relative = val_pct / (val_pct + test_pct)
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, y_temp,
        test_size=1 - val_pct_relative,
        stratify=y_temp,
        random_state=seed
    )

    # Build 3 independent datasets with different transforms
    base_ds_train = datasets.ImageFolder(root=data_loc, transform=train_transform)
    base_ds_val   = datasets.ImageFolder(root=data_loc, transform=val_test_transform)
    base_ds_test  = datasets.ImageFolder(root=data_loc, transform=val_test_transform)

    # Subsets with different transforms
    train_ds = Subset(base_ds_train, train_idx)
    val_ds   = Subset(base_ds_val, val_idx)
    test_ds  = Subset(base_ds_test, test_idx)

    # Compute class weights for oversampling
    class_sample_count = np.bincount(y_train)
    class_weights = 1. / class_sample_count
    sample_weights = class_weights[y_train]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(train_ds, 
                              batch_size=batch_size, 
                              sampler=sampler, 
                              num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds, 
                              batch_size=batch_size, 
                              shuffle=False, 
                              num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds, 
                              batch_size=batch_size, 
                              shuffle=False, 
                              num_workers=num_workers,
                              pin_memory=True)

    return train_loader, val_loader, test_loader, base_ds_train.classes
'''

class CustomTestDataset(Dataset):
    """
    Customized dataset: get label from the file name rather than the folder it is in
    file name is like: <class_name><id>.<ext>
    """
    def __init__(self, root, classes, transform=None):
        self.root_dir = root
        self.transform = transform
        # Class to index
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        # Take all image files
        self.samples = [
            os.path.join(root, fname)
            for fname in os.listdir(root)
            if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        self.prefix_re = re.compile(r'^([^\d]+)')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Read image
        img_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        # Get name of the image
        fname = os.path.basename(img_path)
        m = self.prefix_re.match(fname)
        if not m:
            raise ValueError(f"Cannot extract class from file name: {fname}")
        label_str = m.group(1)  # e.g. 'blight' from 'blight001.jpg'
        
        if label_str not in self.class_to_idx:
            raise ValueError(f"Unknown class '{label_str}' in file {fname}")
        label = self.class_to_idx[label_str]

        # 3. 应用变换
        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_data_loader(batch_size, seed=1000, num_workers=0):
    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=30),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.9, contrast=0.9, saturation=0.9, hue=0.1),
        transforms.RandomAffine(degrees=0, scale=(0.8, 1.0)),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 0.2))], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Build 3 independent datasets with different transforms
    train_loc = r'/Users/mashilin/Desktop/study/coding/APS360/project/clean_dataset/train'
    val_loc = r'/Users/mashilin/Desktop/study/coding/APS360/project/clean_dataset/val'
    test_loc = r'/Users/mashilin/Desktop/study/coding/APS360/project/clean_dataset/test'

    train_ds = datasets.ImageFolder(root=train_loc, transform=train_transform)
    classes = train_ds.classes
    val_ds   = datasets.ImageFolder(root=val_loc, transform=val_test_transform)
    test_ds  = CustomTestDataset(root=test_loc, classes=classes, transform=val_test_transform)

    # Take the number of samples in each class
    class_counts = np.bincount(train_ds.targets)
    # Calculate the weight for each class
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[t] for t in train_ds.targets]
    # Transfer to tensor
    sample_weights = torch.DoubleTensor(sample_weights)

    # Construct samplier
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    # DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,       # Use samplier for including weights
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, val_loader, test_loader, train_ds.classes



#cnn baseline model

# class CNNClassifier(nn.Module):
#     def __init__(self, fc_units=512, num_classes=15):
#         super(CNNClassifier, self).__init__()
#
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.pool2 = nn.MaxPool2d(2, 2)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.pool3 = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(128 * 28 * 28, fc_units)
#         self.fc2 = nn.Linear(fc_units, num_classes)
#
#     def forward(self, x):
#         x = self.pool1(F.relu(self.conv1(x)))
#         x = self.pool2(F.relu(self.conv2(x)))
#         x = self.pool3(F.relu(self.conv3(x)))
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x


__all__ = ['ResNet', 'resnet18']


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
#################################Use Residual Block for Improving Exploding Gradient#################################
# class SEBlock(nn.Module):
#     def __init__(self, channels, r=16):
#         super().__init__()
#         self.fc = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channels, channels // r, 1, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels // r, channels, 1, bias=True),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         return x * self.fc(x)

##########################Inception Block (concat 1x1, 3x3, 5x5 pooling to increase features)##########################
# class InceptionBlock(nn.Module):
#     def __init__(self, in_ch, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
#         super().__init__()
#         # 1×1
#         self.branch1 = nn.Conv2d(in_ch, ch1x1, 1)
#         # 3×3
#         self.branch2 = nn.Sequential(
#             nn.Conv2d(in_ch, ch3x3red, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(ch3x3red, ch3x3, 3, padding=1),
#         )
#         # 5×5
#         self.branch3 = nn.Sequential(
#             nn.Conv2d(in_ch, ch5x5red, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(ch5x5red, ch5x5, 5, padding=2),
#         )
#         # Pooling
#         self.branch4 = nn.Sequential(
#             nn.MaxPool2d(3, stride=1, padding=1),
#             nn.Conv2d(in_ch, pool_proj, 1),
#         )
#     def forward(self, x):
#         return torch.cat([
#             self.branch1(x),
#             self.branch2(x),
#             self.branch3(x),
#             self.branch4(x),
#         ], 1)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))


class CBAM(nn.Module):
    def __init__(self, channels, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, r=16):
        super().__init__()
        norm_layer = norm_layer or nn.BatchNorm2d

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1   = norm_layer(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2   = norm_layer(planes)

        # self.se    = SEBlock(planes, r)
        self.attention = CBAM(planes, r)

        self.downsample = downsample
        self.stride     = stride

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # out = self.se(out)
        out = self.attention(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, norm_layer=None):
        super().__init__()
        norm_layer = norm_layer or nn.BatchNorm2d
        self.inplanes = 64

        self.conv1   = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1     = norm_layer(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = self._make_layer(block,  64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc      = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = nn.BatchNorm2d
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

def _resnet(arch, block, layers, pretrained, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model

def resnet18(pretrained=False, **kwargs):
    return _resnet('resnet18', BasicBlock, [2,2,2,2], pretrained, **kwargs)

import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_ch=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        # 用一个大卷积实现切块 + flatten + 映射
        self.proj = nn.Conv2d(in_ch, embed_dim, 
                              kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        # x: (B,3,H,W)
        x = self.proj(x)           # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)           # (B, embed_dim, N_patches)
        return x.transpose(1, 2)   # (B, N_patches, embed_dim)

# class VisionTransformer(nn.Module):
#     def __init__(self, *, img_size=224, patch_size=16,
#                  in_ch=3, embed_dim=768, depth=12, 
#                  num_heads=12, mlp_ratio=4.0, num_classes=15,
#                  drop_rate=0.0):
#         super().__init__()
#         self.patch_embed = PatchEmbed(img_size, patch_size, in_ch, embed_dim)
#         num_patches = self.patch_embed.num_patches

#         # Learnable [CLS] token + Position Embedding
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
#         self.pos_drop  = nn.Dropout(drop_rate)

#         # TransformerEncoder
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=embed_dim, nhead=num_heads,
#             dim_feedforward=int(embed_dim * mlp_ratio),
#             dropout=drop_rate, activation='gelu'
#         )
#         self.encoder = nn.TransformerEncoder(
#             encoder_layer, num_layers=depth
#         )

#         # 分类头
#         self.norm = nn.LayerNorm(embed_dim)
#         self.head = nn.Linear(embed_dim, num_classes)

#         # 初始化
#         nn.init.trunc_normal_(self.pos_embed, std=0.02)
#         nn.init.trunc_normal_(self.cls_token, std=0.02)
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             nn.init.trunc_normal_(m.weight, std=0.02)
#             if m.bias is not None:
#                 nn.init.zeros_(m.bias)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.zeros_(m.bias)
#             nn.init.ones_(m.weight)

#     def forward(self, x):
#         B = x.size(0)
#         x = self.patch_embed(x)                      # (B, N, D)
#         cls_tokens = self.cls_token.expand(B, -1, -1)# (B,1,D)
#         x = torch.cat((cls_tokens, x), dim=1)        # (B,1+N,D)
#         x = x + self.pos_embed
#         x = self.pos_drop(x).transpose(0,1)          # Transformer 要 (N_seq,B,D)
#         x = self.encoder(x)                          # (1+N,B,D)
#         x = x.transpose(0,1)                         # (B,1+N,D)
#         cls_out = self.norm(x[:,0])                  # 取 CLS token
#         return self.head(cls_out)

def train_net(model, train_loader, val_loader, num_epochs, learning_rate,
              batch_size, metric_prefix="run1", save_path="best_model.pt"):

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(device)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = 0.0
    train_err_list, val_err_list = [], []
    train_loss_list, val_loss_list = [], []

    for epoch in range(num_epochs):

        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_err = 1 - train_acc
        avg_train_loss = running_loss / len(train_loader)


        model.eval()
        correct, total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        val_err = 1 - val_acc
        avg_val_loss = val_loss / len(val_loader)

        # into list
        train_err_list.append(train_err)
        val_err_list.append(val_err)
        train_loss_list.append(avg_train_loss)
        val_loss_list.append(avg_val_loss)

        print(f"[{epoch+1}/{num_epochs}] "
              f"Train Loss={avg_train_loss:.4f}, Train Acc={train_acc*100:.2f}% | "
              f"Val  Loss={avg_val_loss:.4f}, Val  Acc={val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"→ Saved best model (epoch {epoch+1})")
            embedding_list, label_list = [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)

                    x = model.relu(model.bn1(model.conv1(images)))
                    x = model.maxpool(x)

                    x = model.layer1(x)
                    x = model.layer2(x)
                    x = model.layer3(x)
                    x = model.layer4(x)

                    x = model.avgpool(x)
                    feats = torch.flatten(x, 1).cpu().numpy()

                    embedding_list.append(feats)
                    label_list.append(labels.cpu().numpy())

            embeddings = np.concatenate(embedding_list, axis=0)
            labels = np.concatenate(label_list, axis=0)

            np.savetxt("val_features.tsv", embeddings, delimiter='\t')
            np.savetxt("val_labels.tsv", labels.reshape(-1, 1), fmt='%d', delimiter='\t')



    np.savetxt(f"{metric_prefix}_train_err.csv",  np.array(train_err_list))
    np.savetxt(f"{metric_prefix}_val_err.csv",    np.array(val_err_list))
    np.savetxt(f"{metric_prefix}_train_loss.csv",np.array(train_loss_list))
    np.savetxt(f"{metric_prefix}_val_loss.csv",  np.array(val_loss_list))

    return train_err_list, val_err_list, train_loss_list, val_loss_list


def plot_training_curve(path):
    """ Plots the training curve for a model run, given the csv files
    containing the train/validation error/loss.

    Args:
        path: The base path of the csv files produced during training
    """
    train_err = np.loadtxt("{}_train_err.csv".format(path))
    val_err = np.loadtxt("{}_val_err.csv".format(path))
    train_loss = np.loadtxt("{}_train_loss.csv".format(path))
    val_loss = np.loadtxt("{}_val_loss.csv".format(path))
    plt.title("Train vs Validation Error")
    n = len(train_err) # number of epochs
    plt.plot(range(1,n+1), train_err, label="Train")
    plt.plot(range(1,n+1), val_err, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend(loc='best')
    plt.show()
    plt.title("Train vs Validation Loss")
    plt.plot(range(1,n+1), train_loss, label="Train")
    plt.plot(range(1,n+1), val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    # Load Data
    train_loader, val_loader, test_loader, classes = get_data_loader(
        batch_size=128, 
        seed=1000, 
        num_workers=2)

    print("Classes:", classes)
    print("Train batches:", len(train_loader))
    print("Val   batches:", len(val_loader))
    print("Test  batches:", len(test_loader))
    total_num_classes = len(classes)

    # cnn train code
    mode = resnet18(num_classes=total_num_classes)
    train_err, val_err, train_loss, val_loss = train_net(
        mode,
        train_loader,
        val_loader,
        num_epochs=20,
        learning_rate=0.001,
        batch_size=128,
        metric_prefix="Primary model",
        save_path="best_primary_model.pt"
    )
    plot_training_curve("Primary model")


    mode.load_state_dict(torch.load("best_primary_model.pt"))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    mode.to(device)
    mode.eval()

    y_true_test = []
    y_pred_test = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = mode(images)
            _, preds = outputs.max(1)

            y_true_test.extend(labels.cpu().numpy())
            y_pred_test.extend(preds.cpu().numpy())

    # Confusion Matrix and Report
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    cm = confusion_matrix(y_true_test, y_pred_test)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.title("Confusion Matrix - Primary model on Test Set")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    print("Test Accuracy:", accuracy_score(y_true_test, y_pred_test))
    print(classification_report(y_true_test, y_pred_test, target_names=classes))

