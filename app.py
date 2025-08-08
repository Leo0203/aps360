import os
import torch
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import torchvision.transforms as transforms
from project_model.yufer_model import resnet18, get_data_loader
import torch.nn.functional as F
from project_model.is_leaf_model import is_leaf

# Define Flask
app = Flask(__name__, 
            static_folder='static',    # static folder for index
            static_url_path='')        # get index.html
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Front page, return index.html
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

# Accept upload and make prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Protection for upload files
    if 'file' not in request.files:
        return jsonify({'error': 'No upload detected'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Upload file name is empty'}), 400

    # Save to upload folder
    os.makedirs('uploads', exist_ok=True)
    upload_path = os.path.join('uploads', file.filename)
    file.save(upload_path)
    # Determine if this is a leaf
    ok, conf = is_leaf(upload_path, threshold=0.6)
    # Apply the model
    disease, conf = model_predict(upload_path, thresh=0.7)
    if not ok:
        return jsonify({
            'disease': None,
            'confidence': conf,
            'message': 'Image not recognized as a valid leaf.'
        })
    return jsonify({'disease': disease, 'confidence': conf})

# Start the web
if __name__ == '__main__':
    # debug=True for adjustments, false when inference
    app.run(host='0.0.0.0', port=5000, debug=True)
