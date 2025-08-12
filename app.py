import os
import torch
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import torchvision.transforms as transforms
from project_model.yufer_model import resnet18, get_data_loader
import torch.nn.functional as F
from project_model.is_leaf_model import is_leaf
from project_model.result import load_ensemble, predict_image, classes


# Define Flask
app = Flask(__name__, 
            static_folder='static',    # static folder for index
            static_url_path='')        # get index.html
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''compatable with yufer_model
# Get all possible classes
_, _, _, classes = get_data_loader(batch_size=1, num_workers=0)
num_classes = len(classes)

model = resnet18(num_classes=num_classes)
checkpoint = torch.load("best_primary_model.pt", map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std =[0.229,0.224,0.225])
])

def model_predict(image_path, thresh=0.95):
    img = Image.open(image_path).convert('RGB')
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
        top_prob, pred_idx = torch.max(probs, dim=1)
        confidence = top_prob.item()
    if confidence < thresh:
        return None, confidence  
    return classes[pred_idx.item()], confidence
'''



# 三个模型权重（放在 models/ 下，文件名按你的实际修改）
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATHS = [
    os.path.join(MODEL_DIR, 'primary_model_1_seed1000.pt'),
    os.path.join(MODEL_DIR, 'primary_model_2_seed3.pt'),
    os.path.join(MODEL_DIR, 'primary_model_3_seed42.pt'),
]

# 程序启动时加载三模型
try:
    ENSEMBLE = load_ensemble(MODEL_PATHS, device)
except Exception as e:
    print('Model loading error:', e)
    ENSEMBLE = None




# Front page, return index.html
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


'''compatiable with yufer_model
# Accept upload and make prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No upload detected'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Upload file name is empty'}), 400

    os.makedirs('uploads', exist_ok=True)
    upload_path = os.path.join('uploads', file.filename)
    file.save(upload_path)

    # 先做“是否叶片”过滤
    ok, leaf_conf = is_leaf(upload_path, threshold=0.6)
    if not ok:
        return jsonify({
            'disease': None,
            'leaf_confidence': round(float(leaf_conf), 4),
            'message': 'Image not recognized as a valid leaf.'
        })

    # 再做疾病分类（只有是叶片时才运行）
    disease, disease_conf = model_predict(upload_path, thresh=0.7)
    # 如果疾病模型自己因为阈值没通过返回了 None，也一并处理
    if disease is None:
        return jsonify({
            'disease': None,
            'confidence': round(float(disease_conf), 4),
            'message': 'Leaf detected, but disease not confident enough.'
        })

    return jsonify({
        'disease': disease,
        'confidence': round(float(disease_conf), 4)
    })
'''


@app.route('/predict', methods=['POST'])
def predict():
    if ENSEMBLE is None:
        return jsonify({'error': 'Models not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No upload detected'}), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'Upload file name is empty'}), 400

    os.makedirs('uploads', exist_ok=True)
    save_path = os.path.join('uploads', f.filename)
    f.save(save_path)

    # 1) 先过“是否叶片”的门
    ok, leaf_conf = is_leaf(save_path, threshold=0.6)
    if not ok:
        return jsonify({
            'disease': None,
            'leaf_confidence': round(float(leaf_conf), 4),
            'message': 'Image not recognized as a valid leaf.'
        })

    # 2) 用你的三模型 soft vote 做疾病预测
    try:
        raw_label, conf, display = predict_image(ENSEMBLE, save_path, device)
        return jsonify({
            'disease': raw_label,            # 保持向后兼容
            'display_name': display,         # 前端直接用这个名字显示/搜索
            'confidence': round(float(conf), 4)
        })
    except Exception as e:
        print('Inference error:', e)
        return jsonify({'error': 'inference_failed'}), 500



# # Start the web
# if __name__ == '__main__':
#     # debug=True for adjustments, false when inference
#     app.run(host='0.0.0.0', port=5000, debug=True)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

