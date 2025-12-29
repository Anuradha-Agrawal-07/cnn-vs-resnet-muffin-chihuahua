# Muffin vs Chihuahua – Image Classification (CNN vs ResNet18)

This project explores the classic “muffin vs chihuahua” visual confusion problem using deep learning.  
The goal was to compare a convolutional neural network built from scratch with a transfer-learning approach using ResNet18.

## Dataset
kaggle link : https://www.kaggle.com/datasets/samuelcortinhas/muffin-vs-chihuahua-image-classification/data

Two-class image dataset:

```
data/
  train/
    chihuahua/
    muffin/
  test/
    chihuahua/
    muffin/
```

Images were resized to either **128×128** (scratch CNN) or **224×224** (ResNet18).

## Model 1 — Custom CNN (from scratch)

- 3 convolutional blocks  
- ReLU activations  
- Max-Pooling  
- Batch Normalization  
- Dropout  
- Fully connected classifier
- Trained for 15 epochs  

**Accuracy:** ~92%

## Model 2 — ResNet18 (Transfer Learning)

- Pretrained on ImageNet  
- Final layer replaced with 2-class output  
- Trained for 5 epochs (GPU)

**Accuracy:** ~95%  
**Chihuahua Precision:** 1.00  
**Muffin Recall:** 1.00  

## Evaluation (ResNet18)

Confusion Matrix:
```
[[582  58]
 [  2 542]]
```

Classification Report:
- Chihuahua — Precision: 1.00, Recall: 0.91  
- Muffin — Precision: 0.90, Recall: 1.00  
- Accuracy: **0.95**

## Training Environment

Performed in Google Colab with GPU.  
Libraries: PyTorch, TorchVision, scikit-learn

## Repo Structure

```
.
├── scratch_cnn.ipynb
├── resnet18_transfer.ipynb
└── README.md
```

## Future Work

- Data augmentation  
- Try ResNet34 / EfficientNet  
- Export to ONNX/TorchScript  
- Build a small web demo


## 📬 Contact

If you have ideas, suggestions, or want to collaborate, feel free to open an issue or reach out.
