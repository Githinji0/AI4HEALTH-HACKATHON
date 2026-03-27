import os
import cv2

def load_images(data_dir, categories, img_size):
    data = []

    for category in categories:
        path = os.path.join(data_dir, category)
        label = categories.index(category)

        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                image = cv2.imread(img_path)
                image = cv2.resize(image, (img_size, img_size))
                data.append((image, label))
            except:
                continue

    return data

