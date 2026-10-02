"""Train a small CNN on a class-folder image dataset."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf

from src.data_preprocessing import IMAGE_SIZE, load_datasets, show_samples


def build_model(num_classes, image_size=IMAGE_SIZE):
    """Create a compact CNN; rescaling keeps pixel normalization in-model."""
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomTranslation(0.08, 0.08),
    ], name="training_augmentation")
    inputs = tf.keras.Input(shape=(*image_size, 3))
    x = augmentation(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 255)(x)
    for filters in (32, 64, 128):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def save_training_plots(history, results_dir):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    for metric, title, filename in (("accuracy", "Training and validation accuracy", "accuracy.png"), ("loss", "Training and validation loss", "loss.png")):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history.history[metric], label="Training")
        ax.plot(history.history[f"val_{metric}"], label="Validation")
        ax.set(title=title, xlabel="Epoch", ylabel=metric.capitalize())
        ax.legend()
        fig.tight_layout()
        fig.savefig(results_dir / filename, dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset", help="Root containing train/validation/test")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model", default="models/skin_disease_cnn.keras")
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    datasets, class_names = load_datasets(args.dataset, args.batch_size)
    Path(args.results).mkdir(parents=True, exist_ok=True)
    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    show_samples(datasets["train"], class_names, Path(args.results) / "sample_images.png")
    model = build_model(len(class_names))
    model.summary()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(args.model, monitor="val_loss", save_best_only=True),
    ]
    history = model.fit(datasets["train"], validation_data=datasets["validation"], epochs=args.epochs, callbacks=callbacks)
    model.save(args.model)
    Path(args.model).with_suffix(".classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")
    save_training_plots(history, args.results)
    print(f"Saved model to {args.model}")
    print(f"Class order: {class_names}")


if __name__ == "__main__":
    main()
