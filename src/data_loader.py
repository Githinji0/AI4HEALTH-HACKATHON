import os
import cv2

def load_images(data_dir, categories, img_size):
    data = []

    data_dir = os.path.normpath(os.path.expanduser(data_dir))
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Data directory does not exist: {data_dir}. ``DATA_DIR`` must point to your downloaded cell_images dataset."
        )

    for category in categories:
        path = os.path.join(data_dir, category)
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"Category directory does not exist: {path}. Check DATA_DIR and CATEGORIES values."
            )

        label = categories.index(category)

        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                image = cv2.imread(img_path)
                image = cv2.resize(image, (img_size, img_size))
                data.append((image, label))
            except Exception:
                continue

    return data

