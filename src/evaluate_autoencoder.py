import os
import sys
import torch
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, roc_auc_score

sys.path.append(os.path.dirname(__file__))
from train_autoencoder import ConvAutoencoder, IMG_SIZE, transform

MODEL_PATH = "models/autoencoder.pth"
TEST_DIR = "data/bottle/test"


def load_model():
    model = ConvAutoencoder()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model


def reconstruction_error(model, image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        reconstructed = model(tensor)
    error_map = (tensor - reconstructed) ** 2
    error = torch.quantile(error_map, 0.99).item()
    return error


def main():
    model = load_model()

    scores, labels = [], []
    for category in os.listdir(TEST_DIR):
        cat_dir = os.path.join(TEST_DIR, category)
        is_defective = category != "good"
        for fname in os.listdir(cat_dir):
            path = os.path.join(cat_dir, fname)
            error = reconstruction_error(model, path)
            scores.append(error)
            labels.append(1 if is_defective else 0)

    scores = np.array(scores)
    labels = np.array(labels)

    print(f"Total test images: {len(labels)} ({labels.sum()} defective, {len(labels) - labels.sum()} good)")

    auc = roc_auc_score(labels, scores)
    print(f"ROC-AUC (defective vs good, by reconstruction error): {auc:.4f}")

    good_scores = scores[labels == 0]
    threshold = good_scores.mean() + 2 * good_scores.std()
    print(f"Threshold (from good images' error distribution): {threshold:.6f}")

    predictions = (scores > threshold).astype(int)
    print("\nClassification report (0=good, 1=defective):")
    print(classification_report(labels, predictions))


if __name__ == "__main__":
    main()