"""Evaluate a saved model on the held-out test folder."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score

from src.data_preprocessing import load_datasets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--model", default="models/skin_disease_cnn.keras")
    parser.add_argument("--results", default="results")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    datasets, class_names = load_datasets(args.dataset, args.batch_size)
    model = tf.keras.models.load_model(args.model)
    y_true, y_prob = [], []
    for images, labels in datasets["test"]:
        y_true.extend(labels.numpy().tolist())
        y_prob.extend(model.predict(images, verbose=0).tolist())
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_prob).argmax(axis=1)
    labels = list(range(len(class_names)))
    report = classification_report(y_true, y_pred, labels=labels, target_names=class_names, zero_division=0)
    print(f"Test accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro precision: {precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0):.4f}")
    print(f"Macro recall: {recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0):.4f}")
    print(f"Macro F1-score: {f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0):.4f}")
    print("\nClassification report:\n" + report)
    results = Path(args.results)
    results.mkdir(parents=True, exist_ok=True)
    (results / "classification_report.txt").write_text(report, encoding="utf-8")
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    display = ConfusionMatrixDisplay(matrix, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names) * 0.7)))
    display.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False, values_format="d")
    ax.set_title("Test set confusion matrix")
    fig.tight_layout()
    fig.savefig(results / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Show a small batch of test predictions alongside their true labels.
    images, labels_batch = next(iter(datasets["test"]))
    probabilities = model.predict(images, verbose=0)
    count = min(9, int(images.shape[0]))
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < count:
            true_index = int(labels_batch[i])
            pred_index = int(np.argmax(probabilities[i]))
            ax.imshow(tf.cast(images[i], tf.uint8).numpy())
            ax.set_title(f"True: {class_names[true_index]}\nPred: {class_names[pred_index]} ({probabilities[i][pred_index]:.0%})",
                         color="green" if true_index == pred_index else "red", fontsize=9)
    fig.suptitle("Sample test predictions (green = correct, red = incorrect)")
    fig.tight_layout()
    fig.savefig(results / "sample_predictions.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
