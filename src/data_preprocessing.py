"""Dataset splitting, image loading, and preview helpers."""
from pathlib import Path
import random
import shutil
import argparse

import matplotlib.pyplot as plt
import tensorflow as tf

IMAGE_SIZE = (128, 128)
SEED = 42
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def split_class_folders(source_dir, output_dir, train=0.70, validation=0.15, seed=SEED):
    """Copy class subfolders into train/validation/test splits.

    Expected input: source_dir/<class_name>/<image files>. Existing output is
    rejected to avoid silently mixing old and new splits.
    """
    if train <= 0 or validation <= 0 or train + validation >= 1:
        raise ValueError("train and validation must be positive and sum to less than 1")
    source, output = Path(source_dir), Path(output_dir)
    classes = sorted(p for p in source.iterdir() if p.is_dir()) if source.is_dir() else []
    if not classes:
        raise FileNotFoundError(f"No class folders found in {source}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}. Move or remove it first.")
    rng = random.Random(seed)
    for class_dir in classes:
        images = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
        if len(images) < 3:
            raise ValueError(f"Class '{class_dir.name}' needs at least 3 images; found {len(images)}")
        rng.shuffle(images)
        train_end = max(1, int(len(images) * train))
        val_end = max(train_end + 1, int(len(images) * (train + validation)))
        val_end = min(val_end, len(images) - 1)
        for split, paths in (("train", images[:train_end]), ("validation", images[train_end:val_end]), ("test", images[val_end:])):
            destination = output / split / class_dir.name
            destination.mkdir(parents=True, exist_ok=True)
            for path in paths:
                shutil.copy2(path, destination / path.name)
    return [p.name for p in classes]


def load_datasets(dataset_dir, batch_size=32, image_size=IMAGE_SIZE, seed=SEED):
    """Load the explicit train, validation, and test folders."""
    root = Path(dataset_dir)
    class_names = None
    datasets = {}
    for split in ("train", "validation", "test"):
        folder = root / split
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing {split} directory: {folder}")
        ds = tf.keras.utils.image_dataset_from_directory(
            folder, labels="inferred", label_mode="int", image_size=image_size,
            batch_size=batch_size, shuffle=(split == "train"), seed=seed,
        )
        if class_names is None:
            class_names = ds.class_names
        elif ds.class_names != class_names:
            raise ValueError(f"Class folders differ in {split}: expected {class_names}, got {ds.class_names}")
        datasets[split] = ds
    # Cache and prefetch speed up input without changing labels.
    return {name: ds.prefetch(tf.data.AUTOTUNE) for name, ds in datasets.items()}, class_names


def show_samples(dataset, class_names, output_path=None, count=9):
    """Plot one batch of labeled examples."""
    images, labels = next(iter(dataset))
    count = min(count, int(images.shape[0]))
    columns = 3
    rows = (count + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(10, 3.5 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for i, axis in enumerate(axes):
        axis.axis("off")
        if i < count:
            axis.imshow(tf.cast(images[i], tf.uint8).numpy())
            axis.set_title(class_names[int(labels[i])])
    figure.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Create stratified folder splits from class folders")
    parser.add_argument("source", nargs="?", default="data/raw_by_class")
    parser.add_argument("output", nargs="?", default="dataset")
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--validation", type=float, default=0.15)
    args = parser.parse_args()
    names = split_class_folders(args.source, args.output, args.train, args.validation)
    print(f"Created train/validation/test splits for {len(names)} classes in {args.output}")
    print("Classes:", ", ".join(names))


if __name__ == "__main__":
    main()
