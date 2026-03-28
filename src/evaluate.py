import os
from tensorflow.keras.models import load_model
from src.preprocessing import preprocess_data, split_data
from src.data_loader import load_images
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

DATA_DIR = r"C:\WORK\data\cell_images"
CATEGORIES = ["Parasitized", "Uninfected"]
IMG_SIZE = 64
MODEL_PATH = os.path.join("models", "malaria_model.h5")

def evaluate():
    print("Loading and preprocessing data...")
    data = load_images(DATA_DIR, CATEGORIES, IMG_SIZE)
    X, y = preprocess_data(data)
    _, X_test, _, y_test = split_data(X, y)

    print("Loading model...")
    model = load_model(MODEL_PATH)

    print("Evaluating...")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {acc:.2%}")

    y_pred = (model.predict(X_test) > 0.5).astype(int)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CATEGORIES))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    evaluate()