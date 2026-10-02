# Skin Disease Detection Using a CNN

An educational image-classification project using TensorFlow/Keras and a small convolutional neural network (CNN). Given a labeled skin-lesion image, the model returns one of the classes it learned and a softmax score.

> **Important:** This project is for education and research only. It is not a medical device and must not be used to diagnose, treat, or rule out any condition. Model confidence is not a clinical probability. If you have a skin concern, consult a qualified healthcare professional.

## Problem statement and objective

Skin lesion photographs vary in lighting, scale, skin tone, and capture equipment. This project demonstrates an end-to-end supervised image-classification workflow: prepare labeled images, train a CNN, measure held-out performance, and run a prediction on one image. It does not cover all skin diseases and is not suitable for clinical decision-making.

## Technologies

Python, TensorFlow/Keras, NumPy, Pandas (useful for dataset metadata), Matplotlib, scikit-learn, and Pillow.

## Dataset

Example dataset: **HAM10000**, a public collection of 10,015 dermatoscopic images of pigmented lesions across seven diagnostic categories. See the [Harvard Dataverse record (DOI 10.7910/DVN/DBW86T)](https://doi.org/10.7910/DVN/DBW86T) and its associated paper for dataset details and citation. Review the dataset's terms and citation requirements on the source page before using it. The project does not include dataset images.

Download the image files and metadata from Harvard Dataverse. HAM10000 images are provided across image archives and diagnoses in a metadata CSV, rather than ready-made class folders. Organize images into one folder per diagnosis using the metadata's diagnosis column (`dx`) and each image's `image_id` filename. For example:

```text
data/raw_by_class/
├── akiec/       # actinic keratoses / intraepithelial carcinoma
├── bcc/         # basal cell carcinoma
├── bkl/         # benign keratosis-like lesions
├── df/          # dermatofibroma
├── mel/         # melanoma
├── nv/          # melanocytic nevi
└── vasc/        # vascular lesions
```

The labels are dataset annotations, not independent medical advice. Keep the downloaded data outside version control. This project makes an image-level random split for a simple demonstration. HAM10000 may include multiple images from the same lesion; for scientifically sound evaluation, split by lesion/patient identity to prevent leakage. Do not use the simple split for clinical claims.

## Project layout

```text
skin-disease-detection/
├── dataset/                 # generated splits; ignored by Git
│   ├── train/<class>/
│   ├── validation/<class>/
│   └── test/<class>/
├── data/raw_by_class/       # your local source images; ignored by Git
├── notebooks/               # optional experiments
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/                  # generated model and class labels
├── results/                 # generated plots and report
├── requirements.txt
└── README.md
```

## Workflow and CNN architecture

Images are loaded from class-named directories by Keras, resized to 128×128, batched, and shuffled for training. The model rescales pixel values to `[0, 1]` and applies random horizontal flips, small rotations, zoom, and translation during training only. Its layers are: augmentation → rescaling → three `Conv2D + ReLU + MaxPooling2D` blocks (32, 64, 128 filters) → GlobalAveragePooling → Dense(128, ReLU) → Dropout(0.4) → Dense(number of classes, softmax). It uses Adam and sparse categorical cross-entropy, with early stopping and best-validation-loss checkpointing.

## Installation

From this project directory, create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

TensorFlow installation support depends on your operating system and Python version. If pip cannot find a compatible build, use a Python version supported by the TensorFlow release for your platform.

## Prepare the dataset

1. Download HAM10000 and read its usage terms/citation information.
2. Put images into `data/raw_by_class/<diagnosis>/` using the metadata labels and image IDs described above. Each class folder must contain at least three images.
3. Generate deterministic 70/15/15 image-level splits:

```bash
python -m src.data_preprocessing
```

Alternatively, place already-split images in `dataset/train`, `dataset/validation`, and `dataset/test`, each with identical class subfolders. The class folder names become model labels. Do not reuse the test set during tuning.

## Train

```bash
python -m src.train --dataset dataset --epochs 30 --batch-size 32
```

The best validation checkpoint is saved to `models/skin_disease_cnn.keras`; class names are saved alongside as `models/skin_disease_cnn.classes.txt`. Training also creates `results/sample_images.png`, `results/accuracy.png`, and `results/loss.png`. Adjust epochs and batch size for your hardware.

## Evaluate

```bash
python -m src.evaluate --dataset dataset --model models/skin_disease_cnn.keras
```

Evaluation reports test accuracy, macro precision, macro recall, macro F1-score, and per-class precision/recall/F1/support. It saves `results/classification_report.txt`, `results/confusion_matrix.png`, and `results/sample_predictions.png`. Per-class performance matters, especially when dataset categories are imbalanced.

## Predict one image

```bash
python -m src.predict path/to/skin_image.jpg
```

Example format (your result will depend on the training data and run):

```text
Prediction: nv
Confidence: 94.25%
Educational estimate only; not a medical diagnosis.
```

Prediction accepts common JPEG, PNG, and BMP image formats. It gives a clear error if the image, model, or saved class labels are missing or unreadable.

## Results

No accuracy is claimed here because results depend on the data split, hardware, and training run. Train and evaluate the model to populate the plots and report. A random image-level split can overstate performance when related images cross split boundaries, so use lesion/patient-grouped splitting for research comparisons.

## Limitations

- HAM10000 contains selected dermatoscopic images of pigmented lesions, not every skin disease or ordinary phone-camera photos.
- The dataset has class imbalance and demographic/domain limitations; the model may perform unevenly across groups and settings.
- A softmax confidence score is not calibrated diagnostic certainty.
- Image-level splitting can leak lesion-level information when the same lesion has multiple images.
- This model is an educational example only and must not guide medical decisions.

## Future improvements

Use lesion/patient-grouped splits, evaluate on an independent external dataset, quantify uncertainty and calibration, inspect class-wise errors and bias, compare with transfer learning, and add a carefully documented data card and model card.
