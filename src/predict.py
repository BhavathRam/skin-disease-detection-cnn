"""Predict a class and confidence for one image (educational use only)."""
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.data_preprocessing import IMAGE_SIZE


def predict_image(image_path, model_path="models/skin_disease_cnn.keras"):
    path, model_file = Path(image_path), Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {path}")
    if not model_file.is_file():
        raise FileNotFoundError(f"Trained model not found: {model_file}. Train the model first.")
    class_file = model_file.with_suffix(".classes.txt")
    if not class_file.is_file():
        raise FileNotFoundError(f"Class label file not found: {class_file}. Train the model first.")
    try:
        image = tf.keras.utils.load_img(path, target_size=IMAGE_SIZE)
        array = tf.keras.utils.img_to_array(image)
    except Exception as exc:
        raise ValueError(f"Could not read image '{path}': {exc}") from exc
    model = tf.keras.models.load_model(model_file)
    probabilities = model.predict(np.expand_dims(array, axis=0), verbose=0)[0]
    class_names = class_file.read_text(encoding="utf-8").splitlines()
    if len(class_names) != len(probabilities):
        raise ValueError("Saved class labels do not match model output dimensions")
    index = int(np.argmax(probabilities))
    return class_names[index], float(probabilities[index])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to an image file")
    parser.add_argument("--model", default="models/skin_disease_cnn.keras")
    args = parser.parse_args()
    try:
        label, confidence = predict_image(args.image, args.model)
        print(f"Prediction: {label}\nConfidence: {confidence * 100:.2f}%")
        print("Educational estimate only; not a medical diagnosis.")
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
