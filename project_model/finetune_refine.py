import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split


DATA_ROOT = r'C:\\Users\\16482\\Desktop\\aps360\\cleandata\\cleandata'
BATCH_SIZE = 128
SEED = 1000
VAL_PCT = 0.1
TEST_PCT = 0.1
FINETUNE_EPOCHS = 5
LR = 1e-4
WEIGHT_DECAY = 1e-4
BEST_PATH_IN = 'best_primary_model.pt'
BEST_PATH_OUT = 'best_primary_model_refined.pt'

# 加速
torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

# -----------------------------
# 数据加载
# -----------------------------
def get_data_loader(data_loc, batch_size, val_pct=0.1, test_pct=0.1, seed=1000, num_workers=2):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    full_ds_plain = datasets.ImageFolder(root=data_loc)
    targets = np.array(full_ds_plain.targets)

    train_idx, temp_idx, y_train, y_temp = train_test_split(
        np.arange(len(targets)), targets,
        test_size=val_pct + test_pct,
        stratify=targets,
        random_state=seed
    )

    val_pct_relative = val_pct / (val_pct + test_pct)
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, y_temp,
        test_size=1 - val_pct_relative,
        stratify=y_temp,
        random_state=seed
    )

    base_ds_train = datasets.ImageFolder(root=data_loc, transform=train_transform)
    base_ds_val   = datasets.ImageFolder(root=data_loc, transform=val_test_transform)
    base_ds_test  = datasets.ImageFolder(root=data_loc, transform=val_test_transform)

    train_ds = Subset(base_ds_train, train_idx)
    val_ds   = Subset(base_ds_val, val_idx)
    test_ds  = Subset(base_ds_test, test_idx)

    class_sample_count = np.bincount(y_train)
    class_weights = 1. / class_sample_count
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    loader_kwargs = dict(num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, base_ds_train.classes

# -----------------------------
# 模型
# -----------------------------

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

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
        return self.sigmoid(self.shared_MLP(self.avg_pool(x)) + self.shared_MLP(self.max_pool(x)))

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
        self.attention = CBAM(planes, r)
        self.downsample = downsample
        self.stride     = stride
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.attention(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=38, norm_layer=None):
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
    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample, norm_layer)]
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

def resnet18(num_classes=38):
    return ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes)

# -----------------------------
# EarlyStopping
# -----------------------------
class EarlyStopping:
    def __init__(self, patience=2, mode='max', min_delta=1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.num_bad = 0
        self.should_stop = False
    def step(self, metric):
        if self.best is None:
            self.best = metric
            return True
        improve = (metric - self.best) > self.min_delta if self.mode == 'max' else (self.best - metric) > self.min_delta
        if improve:
            self.best = metric
            self.num_bad = 0
            return True
        else:
            self.num_bad += 1
            if self.num_bad >= self.patience:
                self.should_stop = True
            return False

# -----------------------------
# 训练/验证/测试
# -----------------------------

def finetune(model, train_loader, val_loader, epochs=5):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda')
    early = EarlyStopping(patience=2, mode='max', min_delta=1e-4)

    best_val = 0.0
    for ep in range(epochs):
        # ---- train ----
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"FT {ep+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            with torch.no_grad():
                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        train_acc = correct / total
        avg_train_loss = running_loss / len(train_loader)

        # ---- val ----
        model.eval()
        correct, total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        avg_val_loss = val_loss / len(val_loader)

        print(f"[FT {ep+1}/{epochs}] Train Loss={avg_train_loss:.4f}, Train Acc={train_acc*100:.2f}% | "
              f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc*100:.2f}%")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), BEST_PATH_OUT)
            print('→ Saved refined best')

        _ = early.step(val_acc)
        if early.should_stop:
            print('Early stopping triggered.')
            break


def evaluate(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print('Refined Test Acc:', correct / total)

# -----------------------------
# main
# -----------------------------
if __name__ == '__main__':
    train_loader, val_loader, test_loader, classes = get_data_loader(
        DATA_ROOT, batch_size=BATCH_SIZE, val_pct=VAL_PCT, test_pct=TEST_PCT, seed=SEED, num_workers=2
    )
    print('Classes:', len(classes))

    # 调用最优模型
    model = resnet18(num_classes=len(classes))
    model.load_state_dict(torch.load(BEST_PATH_IN, map_location=device))
    model.to(device)

    # 微调
    finetune(model, train_loader, val_loader, epochs=FINETUNE_EPOCHS)

    # 评估
    model.load_state_dict(torch.load(BEST_PATH_OUT, map_location=device))
    model.to(device)
    evaluate(model, test_loader)
