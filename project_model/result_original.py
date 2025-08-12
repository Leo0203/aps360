import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from test import resnet18

# 模型文件路径
model_paths = [
    "C:\\Users\\16482\\Desktop\\aps360\\primary_model_1_seed1000.pt",
    "C:\\Users\\16482\\Desktop\\aps360\\primary_model_2_seed3.pt",
    "C:\\Users\\16482\\Desktop\\aps360\\primary_model_3_seed42.pt"
]

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


# 图片路径
image_path = "C:\\Users\\16482\\Desktop\\aps360\\2.jpg"

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

#读取图片
image = Image.open(image_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_tensor = input_tensor.to(device)

#加载模型
models = []
for path in model_paths:
    model = resnet18(num_classes=len(classes))
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    models.append(model)

# 推理
all_preds = []
with torch.no_grad():
    for i, model in enumerate(models, start=1):
        outputs = model(input_tensor)
        _, pred = outputs.max(1)
        pred_class = classes[pred.item()]
        all_preds.append(pred_class)
        print(f"模型{i}预测结果: {pred_class}")

#投票
def soft_vote(logits_list):
    probs = [torch.softmax(l, dim=1) for l in logits_list]
    avg_prob = torch.mean(torch.stack(probs, dim=0), dim=0)
    top1 = torch.argmax(avg_prob, dim=1).item()
    conf = avg_prob[0, top1].item()
    return top1, conf

logits_list, per_model_top1, per_model_conf = [], [], []

with torch.no_grad():
    for i, m in enumerate(models, 1):
        logits = m(input_tensor)
        prob = torch.softmax(logits, dim=1)
        top1 = torch.argmax(prob, dim=1).item()
        conf1 = prob[0, top1].item()

        logits_list.append(logits)
        per_model_top1.append(top1)
        per_model_conf.append(conf1)
        print(f"模型{i} -> {classes[top1]} (conf={conf1:.4f})")

final_id, final_conf = soft_vote(logits_list)
print(f"\nSoft voting 最终结果: {classes[final_id]} (conf={final_conf:.4f})")