import os
from src.data_loader import load_images
from src.preprocessing import preprocess_data, split_data
from src.model import build_cnn

import tensorflow as tf

# Config
DATA_DIR = r"C:\WORK\data\cell_images"
CATEGORIES = ["Parasitized", "Uninfected"]
IMG_SIZE = 64
MODEL_PATH = os.path.join("models", "malaria_model.h5")

def train():
    print("Loading data...")
    data = load_images(DATA_DIR, CATEGORIES, IMG_SIZE)

    print("Preprocessing...")
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Building model...")
    model = build_cnn()

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("Training...")
    model.fit(
        X_train, y_train,
        epochs=10,
        validation_data=(X_test, y_test),
        batch_size=32
    )

    print("Saving model...")
    model.save(MODEL_PATH)

    return model


if __name__ == "__main__":
    train()