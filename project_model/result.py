# project_model/result.py — refactored to be importable by app.py
from typing import List, Tuple
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

try:
    # 如果在包里使用相对导入
    from .test import resnet18
except Exception:
    # 如果和 test.py 在同一目录
    from test import resnet18

# === 确保与训练一致的类别顺序（用你原来的 classes 列表）===
# 类别名称
classes = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]


# === 预处理（与你训练/原脚本一致）===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_ensemble(model_paths: List[str], device: torch.device) -> List[torch.nn.Module]:
    """加载多份权重，返回 eval() 且在 device 上的模型列表"""
    ms: List[torch.nn.Module] = []
    for p in model_paths:
        m = resnet18(num_classes=len(classes))
        state = torch.load(p, map_location=device)
        m.load_state_dict(state)
        m.to(device)
        m.eval()
        ms.append(m)
    return ms

@torch.no_grad()
def predict_image(models: List[torch.nn.Module], image_path: str, device: torch.device
                  ) -> Tuple[str, float, str]:
    """
    对单张图片做软投票，返回：
      raw_label: 原始类别名，如 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
      confidence: 平均 softmax 的 top-1 概率
      display_name: 友好名称（用于前端显示/搜索），如 'Tomato Yellow Leaf Curl Virus'
    """
    img = Image.open(image_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)

    prob_sum = None
    for m in models:
        logits = m(x)
        probs = F.softmax(logits, dim=1)
        prob_sum = probs if prob_sum is None else prob_sum + probs

    avg_prob = prob_sum / len(models)
    top1 = int(torch.argmax(avg_prob, dim=1).item())
    conf = float(avg_prob[0, top1].item())

    raw_label = classes[top1]
    display_name = raw_label.split('___')[-1].replace('_', ' ') if '___' in raw_label else raw_label.replace('_', ' ')
    return raw_label, conf, display_name
