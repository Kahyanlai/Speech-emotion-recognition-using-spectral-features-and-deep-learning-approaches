# 🧠 Speech Emotion Recognition using Spectral Features and Deep Learning

## 📄 Description
This project implements **speech emotion recognition (SER)** using both handcrafted **spectral features** and **deep learning** methods.  

The system classifies speech into emotional categories (*Angry, Calm, Happy, Sad*) by:
- Extracting **Spectral Centroid (SC)**, **Spectral Bandwidth (SBW)**, and **Spectral Band Energy (SBE)** features manually.
- Training **SVM** models on handcrafted features.
- Designing **1D CNN** and **2D CNN** architectures that learn directly from audio waveforms and spectrogram images.

---

## 🎯 Objectives
1. Implement spectral feature extraction manually (without using `librosa.feature` shortcuts).  
2. Train and evaluate an **SVM** on handcrafted spectral features.  
3. Build and compare **1D CNN** and **2D CNN** models using PyTorch.  
4. Analyse performance through accuracy, confusion matrices, and learning curves.

---

## 🧩 Methods and Components

| Component | Description |
|------------|-------------|
| **Spectral Feature Extraction** | Computes SC, SBW, and SBE manually using mel-scale spectrograms (300–3400 Hz). |
| **Dataset Loader** | Custom `EmotionSpeechDataset` class for normalising and padding speech signals. |
| **SVM Model** | RBF-kernel SVM (C = 10) trained on handcrafted features. |
| **1D CNN** | Learns temporal patterns directly from raw waveforms. |
| **2D CNN** | Trains on mel-spectrogram “images” resized to 150×50 pixels. |
| **Evaluation** | Overall accuracy, confusion matrix, and learning curves. |

---

## 📊 Results Summary
- **Best handcrafted feature:** Spectral Band Energy (SBE) achieved highest SVM accuracy.  
- **1D CNN:** Improved recognition accuracy by learning from raw temporal patterns.  
- **2D CNN:** Achieved the best overall performance using spatial-spectral features from mel-spectrograms.

---

## 🧠 Key Insights
- Handcrafted features are strong baselines but deep models generalise better.  
- Convolutional layers can automatically learn rich emotional cues.  
- Consistent input length and proper normalisation are crucial for stable training.

---

## 🛠️ Tech Stack
- **Language:** Python 3.9+  
- **Libraries:** PyTorch • Librosa • NumPy • OpenCV • Matplotlib • Scikit-learn • Pydub  
- **Environment:** Jupyter Notebook / Google Colab
