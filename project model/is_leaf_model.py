import torch
from PIL import Image
import clip

# Define a model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Define judgmental function
def is_leaf(image_path, threshold=0.5):
    """
    使用 CLIP 的 zero-shot 功能判断图片是否为叶子。
    :param image_path: image stored path
    :param threshold: threshold value for is or not a leaf
    :return: (is_leaf: bool, leaf_prob: float)
    """
    image = preprocess(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)
    # Text descriptions
    texts = clip.tokenize(["a photo of a leaf", "not a leaf"]).to(device)

    with torch.no_grad():
        logits_per_image, _ = model(image, texts)
        probs = logits_per_image.softmax(dim=-1)
    leaf_prob = probs[0][0].item()
    return leaf_prob >= threshold, leaf_prob
