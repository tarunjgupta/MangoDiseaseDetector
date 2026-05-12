# 🥭 Patch-Based CNN for Mango Disease Detection

A deep learning–based research project for detecting mango diseases using a **Patch-Based CNN approach**.
The system uses **MobileNetV2 (Transfer Learning)** for classification, **Patch-Based Inference** for better localization, **Grad-CAM** for explainability, and a **Streamlit UI** for interaction.

---

## 📌 Features

* ✅ Transfer Learning using MobileNetV2
* ✅ Patch-Based Prediction (core research idea)
* ✅ Grad-CAM Heatmap for explainability
* ✅ Simple Streamlit Web UI
* ✅ CLI Prediction Support

---

## 📁 Project Structure

```
PatchBasedDetection/
│
├── dataset/                     # Dataset folder
│   ├── Anthracnose/
│   ├── Bacterial_Canker/
│   ├── Scab/
│   └── Healthy/
│
├── saved_model/                # Saved after training
│   ├── mango_model.keras
│   └── class_names.txt
│
├── dataset_loader.py           # Data loading & augmentation
├── model.py                    # MobileNetV2 model
├── train.py                    # Training script
├── patch_utils.py              # Patch extraction + aggregation
├── predict.py                  # CLI prediction
├── gradcam.py                  # Grad-CAM heatmap
├── app.py                      # Streamlit UI
├── requirements.txt
└── README.md
```

---

## 🚀 Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/tarunjgupta/PatchBasedMangoDiseaseDetection.git
cd PatchBasedMangoDiseaseDetection
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Prepare Dataset

```
dataset/
├── Anthracnose/        (150–200 images)
├── Bacterial_Canker/   (150–200 images)
├── Scab/               (150–200 images)
└── Healthy/            (150–200 images)
```

---

### 5️⃣ Train the Model

```bash
python train.py --dataset ./dataset --epochs 10
```

✔ Model saved to:

```
saved_model/mango_model.keras
```

---

### 6️⃣ Test Prediction (CLI)

```bash
python predict.py --image test.jpg
```

---

### 7️⃣ Run Web App

```bash
streamlit run app.py
```

👉 Open: http://localhost:8501

---

## 🧠 How It Works

### 🔹 Training

* Images resized to **224×224**
* MobileNetV2 (ImageNet pretrained)
* Base model frozen
* Custom classifier trained
* Data augmentation applied

---

### 🔹 Patch-Based Prediction (Core Logic)

1. Image → split into **128×128 patches**
2. Each patch → resized to 224×224
3. Model predicts each patch
4. Predictions are aggregated
5. Final class is selected

---

### 🔹 Grad-CAM Explainability

* Highlights important regions
* Shows where the model focuses
* Overlay heatmap on original image

---

## 📊 Classes

| Class            | Description              |
| ---------------- | ------------------------ |
| Anthracnose      | Dark fungal spots        |
| Bacterial Canker | Raised bacterial lesions |
| Scab             | Rough fungal patches     |
| Healthy          | No disease               |

---

## ⚙️ Configuration

| Parameter        | Value   |
| ---------------- | ------- |
| Input Size       | 224×224 |
| Patch Size       | 128×128 |
| Batch Size       | 16      |
| Epochs           | 10      |
| Learning Rate    | 0.001   |
| Validation Split | 20%     |

---

## 🛠 Troubleshooting

| Issue             | Solution          |
| ----------------- | ----------------- |
| Model not loading | Train first       |
| Module error      | Activate venv     |
| Slow training     | Reduce dataset    |
| Memory issue      | Reduce batch size |

---

## 🚀 Future Improvements

* Patch-wise Grad-CAM
* Severity-based patch weighting
* Segmentation-based detection
* Mobile deployment

---

## 📌 Author

**Tarunj Gupta**

---

## ⭐ If you like this project, give it a star!
