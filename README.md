# PatchCore Metal Surface Anomaly Detection

An industrial metal surface inspection system built using the **PatchCore anomaly detection approach**.

The system learns the visual characteristics of **normal metal surfaces** and detects abnormal or defective regions by comparing extracted image features with previously stored normal features.

The project uses a pretrained **ResNet50** for feature extraction, a memory bank for storing normal descriptors, **FAISS** for efficient nearest-neighbor search, and a **Streamlit dashboard** for interactive inspection and visualization.

---

# Project Overview

Traditional defect detection systems usually require labeled examples of every possible defect.

In industrial environments, this can be difficult because:

- Defects can be rare.
- New defect types may appear.
- Collecting labeled defective images is expensive.
- It is difficult to predict every possible failure.

PatchCore solves this problem using **anomaly detection**.

Instead of learning:

> What does every defect look like?

The system learns:

> What does a normal metal surface look like?

During inference, the system compares a new image with the learned normal patterns.

If a region is very different from normal patterns, it receives a high anomaly score.

---

# Project Objective

The objective of this project is to build an automated inspection system capable of:

- Learning normal metal surface patterns.
- Extracting deep features using ResNet50.
- Creating local PatchCore descriptors.
- Building a memory bank from normal images.
- Reducing the memory bank using coreset sampling.
- Building a FAISS nearest-neighbor index.
- Detecting anomalies in new images.
- Generating anomaly heatmaps.
- Classifying images as **GOOD** or **DEFECTIVE**.
- Providing an interactive Streamlit dashboard.

---

# Project Architecture

```text
                         TRAINING PIPELINE

Normal Metal Images
        │
        ▼
Image Preprocessing
        │
        ▼
Pretrained ResNet50
        │
        ├───────────────┐
        ▼               ▼
     Layer2          Layer3
     Features        Features
        │               │
        └───────┬───────┘
                ▼
        Feature Embeddings
                │
                ▼
       PatchCore Descriptors
                │
                ▼
        Random Projection
                │
                ▼
        Coreset Sampling
                │
                ▼
          Memory Bank
                │
                ▼
           FAISS Index
                │
                ▼
      Stored Normal Patterns


🔵 Inference Pipeline

During inference, a new metal surface image follows the same feature extraction pipeline.

New Metal Image
        │
        ▼
Image Preprocessing
        │
        ▼
Pretrained ResNet50
        │
        ├───────────────┐
        ▼               ▼
     Layer2          Layer3
     Features        Features
        │               │
        └───────┬───────┘
                ▼
        PatchCore Descriptors
                │
                ▼
      FAISS Nearest Neighbor Search
                │
                ▼
 Distance from Normal Descriptors
                │
                ▼
        Anomaly Score
                │
        ┌───────┴────────┐
        ▼                ▼
     GOOD            DEFECTIVE
                │
                ▼
          Anomaly Heatmap
                │
                ▼
         Streamlit Dashboard

🧠 Technologies Used

| Technology   | Purpose                      |
| ------------ | ---------------------------- |
| Python       | Main programming language    |
| PyTorch      | Deep learning framework      |
| Torchvision  | Pretrained ResNet50 model    |
| ResNet50     | Feature extraction           |
| FAISS        | Fast nearest-neighbor search |
| NumPy        | Numerical computation        |
| OpenCV       | Image processing             |
| Scikit-learn | Evaluation metrics           |
| Matplotlib   | Visualization                |
| Streamlit    | Web dashboard                |

🧩 Feature Extraction
This project uses a pretrained ResNet50 model as the feature extractor.

Instead of using the final classification output of ResNet50, intermediate feature maps are extracted.

The following layers are used:
ResNet50

Layer 2 ─────► Medium-level features

Layer 3 ─────► High-level features

These layers capture different levels of information about the metal surface.

Layer 2

Captures:

Surface texture
Local patterns
Edges
Small structures

Layer 3

Captures:

More complex patterns
Structural information
Larger visual regions

The features from these layers are combined to create powerful local descriptors.

🧠 What Are Descriptors?

A descriptor is a numerical representation of a small region of an image.

For example:
Metal Surface Image
        │
        ▼
Deep Feature Extraction
        │
        ▼
Descriptor 1 → [0.21, 0.42, 0.81, ...]
Descriptor 2 → [0.53, 0.19, 0.73, ...]
Descriptor 3 → [0.14, 0.62, 0.29, ...]

Each descriptor represents the visual characteristics of a local region.

Instead of comparing entire images, PatchCore compares these local descriptors.

This helps detect small defects.

🗂️ Memory Bank

After processing all normal training images, their descriptors are collected.
Normal Image 1 ──► Descriptors
Normal Image 2 ──► Descriptors
Normal Image 3 ──► Descriptors
Normal Image N ──► Descriptors
                        │
                        ▼
                  Memory Bank

The memory bank represents:What a normal metal surface looks like.

✂️ Coreset Sampling

A complete memory bank can contain a very large number of descriptors.

Searching through all descriptors would be slow and memory-intensive.

Therefore, coreset sampling is used.

Large Memory Bank
       │
       ▼
Coreset Sampling
       │
       ▼
Representative Descriptors
       │
       ▼
Smaller Memory Bank

The goal is to keep descriptors that represent the normal data distribution while reducing the total number of stored descriptors.

⚡ FAISS Index

The final normal descriptors are stored inside a FAISS index.

FAISS allows fast nearest-neighbor search.

Normal Descriptors
        │
        ▼
   FAISS Index
        │
        ▼
Fast Similarity Search

During inference, every descriptor from a new image is compared against the stored normal descriptors.

FAISS finds the closest normal descriptor.

📏 Anomaly Detection

Suppose a new image produces a descriptor:

New Descriptor
      │
      ▼
FAISS Search
      │
      ▼
Closest Normal Descriptor
      │
      ▼
Calculate Distance

The distance represents how different the new region is from the normal patterns.

Small Distance
New Region ≈ Normal Region

Result:

LOW ANOMALY
Large Distance
New Region ≠ Normal Region

Result:

HIGH ANOMALY
📊 Image-Level Anomaly Score

Each local descriptor receives an anomaly distance.

For example:

Descriptor 1 → Distance = 5.2
Descriptor 2 → Distance = 8.7
Descriptor 3 → Distance = 3.1
Descriptor 4 → Distance = 29.4
Descriptor 5 → Distance = 6.8

The image-level anomaly score is calculated from the highest anomaly value:

Image Score = max(local anomaly distances)

Therefore:

Image Score = 29.4

This makes sense because even a small defect can indicate that the entire image should be considered defective.

🎯 Threshold Selection

A threshold is used to convert the anomaly score into a final prediction.

If Score < Threshold
        │
        ▼
       GOOD
If Score ≥ Threshold
        │
        ▼
    DEFECTIVE

The threshold is automatically calculated using the ROC curve.

The project uses:

Youden's J Statistic

J = TPR - FPR

The threshold that maximizes:

TPR - FPR

is selected as the recommended threshold.

📈 Model Evaluation

The model is evaluated using the following metrics:

Accuracy

Measures overall correct predictions.

Accuracy =
Correct Predictions
-------------------
Total Predictions
Precision

Measures how many predicted defects are actually defective.

Recall

Measures how many actual defects were successfully detected.

F1 Score

Balances precision and recall.

AUROC

Measures how well the model separates:

GOOD Images

from:

DEFECTIVE Images

A higher AUROC indicates better separation.

📊 Example Evaluation Results

The model achieved the following results during evaluation:

AUROC     : 0.9965

Accuracy  : 0.9827
Precision : 0.9972
Recall    : 0.9841
F1 Score  : 0.9906

Confusion Matrix:

[[  84    3]
 [  17 1052]]

This indicates strong performance in distinguishing normal and defective metal surfaces.

🔥 Anomaly Heatmap

The local anomaly distances are converted into an anomaly map.

Low Anomaly ───────► Blue / Cool Regions

High Anomaly ──────► Red / Hot Regions

The heatmap helps visualize the areas where the model detects unusual patterns.

The dashboard displays:

Original Image

        +

Anomaly Heatmap

        +

Defect Overlay

🖥️ Streamlit Dashboard

The project includes an interactive Streamlit dashboard.

The dashboard allows users to:

Upload a metal surface image
Run PatchCore inference
View the prediction
View the anomaly score
View the threshold
View the original image
View the anomaly heatmap
View the defect overlay
Download the inspection report

🚀 Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/Patchcore-metal-detection.git

Move into the project directory:

cd Patchcore-metal-detection

Create a virtual environment:

python -m venv .venv

Activate the environment.

Windows
.venv\Scripts\activate
Linux / macOS
source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

🏋️ Build the Memory Bank

Place normal training images inside:

data/train/good/

Then run the memory bank generation pipeline.

The process is:

Normal Images
      │
      ▼
Feature Extraction
      │
      ▼
Descriptors
      │
      ▼
Memory Bank
      │
      ▼
Coreset Sampling
      │
      ▼
FAISS Index

The generated FAISS index is stored inside:

outputs/index.faiss

🧪 Evaluate the Model

Run:

cd src/inference
python evaluate_model.py

The evaluation script calculates:

AUROC
Accuracy
Precision
Recall
F1 Score
Confusion Matrix
Recommended Threshold

It also generates:

outputs/roc_curve.png

🖥️ Run the Dashboard

From the project root:

streamlit run app/app.py

The dashboard will open in your browser.

Usually at:

http://localhost:8501

👩‍💻 Author
Janhavi Lakeri
Computer Vision Project

⭐ Project Summary

This project implements an end-to-end PatchCore-based industrial anomaly detection system for metal surface inspection.

The system learns what a normal surface looks like and detects defects by measuring how different new image regions are from previously learned normal patterns.

Learn Normal Patterns
        +
Compare New Images
        +
Measure Differences
        =
Detect Anomalies