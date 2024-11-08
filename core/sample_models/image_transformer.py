import torch
from transformers import ViTModel, ViTFeatureExtractor
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained ViT model and feature extractor
model_name = "google/vit-base-patch16-224"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTModel.from_pretrained(model_name)


def preprocess_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs


def extract_features(image_path):
    # Preprocess the image
    inputs = preprocess_image(image_path)
    # Extract features
    with torch.no_grad():
        outputs = model(**inputs)
    # Extract the [CLS] token embeddings
    features = outputs.last_hidden_state[:, 0, :].numpy()
    return features


def compute_similarity(features1, features2):
    # Compute cosine similarity between two sets of features
    similarity = cosine_similarity(features1, features2)
    return similarity
