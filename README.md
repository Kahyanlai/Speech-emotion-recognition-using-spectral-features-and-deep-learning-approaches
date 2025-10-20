# 🧠 Speech Emotion Recognition using Spectral Features and Deep Learning

## 📘 Overview
This project investigates how both **handcrafted spectral features** and **deep neural networks** can be applied to recognise human emotions from speech.  
It was completed as part of **SIT789 – Robotics, Computer Vision and Speech Processing (Task 8.2D)**.

The study implements and compares three main approaches:
1. Manual extraction of **Spectral Centroid (SC)**, **Spectral Bandwidth (SBW)**, and **Spectral Band Energy (SBE)**.
2. **SVM** classification using handcrafted spectral features.
3. **1D CNN** and **2D CNN** architectures trained directly on raw waveforms and mel-spectrograms.

---

## 🎯 Objectives
- Implement SC, SBW, SBE manually without `librosa.feature` shortcuts.  
- Train an SVM on handcrafted spectral features as a baseline.  
- Design, train, and compare 1D CNN and 2D CNN models.  
- Evaluate which representations and network types generalise best for emotion recognition.

---

## 🧩 Methodology

### 1️⃣ Spectral Feature Extraction
- Computed **mel-scale spectrograms** (`n_mels = 3401`, `n_fft = 16384`, `hop_length = int(sr × 0.015)`).
- Extracted **SC**, **SBW**, and **SBE** manually over 5 frequency sub-bands (300–3400 Hz, 200 frames each).  
- Saved features to disk for reuse.

🧠 *Purpose:* Provide interpretable acoustic descriptors to benchmark against learned features.

---

### 2️⃣ SVM Baseline
- Model: **SVM (RBF kernel, C = 10)**  
- Input: handcrafted feature vectors  
- Output: four emotion classes (*Angry, Calm, Happy, Sad*)  
- Evaluation: overall accuracy + confusion matrix  

#### 📊 Results
| Feature | Accuracy | Observation |
|----------|-----------|-------------|
| Spectral Centroid (SC) | **51.56 %** | Moderate frequency focus |
| Spectral Bandwidth (SBW) | **47.66 %** | Least stable feature |
| Spectral Band Energy (SBE) | **55.47 %** | Best handcrafted performance |

🧠 *Insight:* SBE captured the most emotion-related information, especially for *Calm* and *Angry* classes.

---

### 3️⃣ 1D CNN – Learning from Raw Waveforms
- Input: normalised 1-D audio signal (200 frames)  
- Architecture:
  - 1D Conv → ReLU → MaxPool → Dropout (× 3 blocks)  
  - Fully Connected → Softmax output  
- Training: Cross-Entropy Loss + Adam optimizer  
- Evaluation: accuracy + confusion matrix + learning curve  

#### 📊 Result
- **Accuracy:** 62.50 %  
- Validation accuracy stabilised ≈ 0.6 with fluctuating validation loss.  
- Strongest recognition for *Angry* and *Calm*; *Sad* remained hardest.  

🧠 *Conclusion:* Learns temporal and energy variations beyond handcrafted features, but lacks spectral abstraction.

---

### 4️⃣ 2D CNN – Learning from Mel-Spectrogram Images
Mel-spectrograms were converted to grayscale “images” (150 × 50), normalised, and used as CNN inputs.

#### 🔹 Baseline 2D CNN (no augmentation)
- Custom 5-layer CNN trained from scratch.  
- **Accuracy:** 68.75 %  
- Overfitting observed (validation plateau ≈ 0.55–0.60).  

#### 🔹 Baseline + Spectrogram Augmentation
- Applied frequency/time masking, amplitude scaling, noise injection, and time shifting.  
- Accuracy dropped to **48.44 %**, indicating unstable learning when trained from random initialisation.

#### 🔹 Improved Model – EfficientNet-B0 with Transfer Learning + Augmentation
To address data scarcity and instability:
- Adopted **pretrained EfficientNet-B0** backbone (ImageNet weights).  
- Applied **two-phase fine-tuning**:  
  - Phase 1 – train classifier + upper conv blocks (LR = 0.0003).  
  - Phase 2 – unfreeze all layers + fine-tune (LR = 0.00005).  
- Regularisation: Dropout (0.4 / 0.3), Mixup, Label Smoothing (0.05).  
- Scheduler: Cosine Annealing LR.  

#### 📊 Final Result
| Model | Accuracy | Key Observations |
|--------|-----------|------------------|
| 2D CNN (no aug) | 68.75 % | Good but overfit |
| 2D CNN (aug only) | 48.44 % | Training instability |
| **2D CNN (EfficientNet-B0 + aug)** | **75.00 %** | Stable training, best generalisation |

- Highest recall for *Happy (93.8 %)* and *Angry (73.7 %)*.  
- *Sad* remained most challenging due to overlapping spectral cues.

🧩 *Outcome:* Transfer learning + domain-specific augmentation yielded + 12.5 % accuracy improvement and significantly more stable validation curves.

---

## 🧠 Insights and Discussion
- **1D CNNs** learn temporal dynamics but lack full spectral context.  
- **2D CNNs** exploit time–frequency relationships, outperforming handcrafted and 1D models.  
- **Augmentation alone** is ineffective on small randomly initialised networks.  
- **Transfer learning** provides pretrained spectral priors, allowing augmentations to improve robustness.  

---

## 💬 Opinion on Data Augmentation
Standard image augmentations (e.g., rotation, flipping) distort temporal–frequency structure and are **not suitable** for spectrograms.  
Instead, **domain-specific methods**—time/frequency masking, amplitude scaling, noise injection—are more effective when combined with pretrained models.  
In this study, augmentation became beneficial **only after** integrating EfficientNet-B0, confirming that effectiveness depends on both **augmentation design** and **network prior knowledge**.

---

## ⚙️ Technical Summary
| Category | Details |
|-----------|----------|
| **Language** | Python 3.9+ |
| **Frameworks** | PyTorch • Librosa • OpenCV • NumPy • Scikit-learn |
| **Training Hardware** | Google Colab GPU |
| **Dataset** | EmotionSpeech (4 classes: Angry, Calm, Happy, Sad) |
| **Metrics** | Accuracy • Confusion Matrix • Learning Curves |

---

## 🚀 How to Run
1. **Install dependencies**
   ```bash
   pip install torch torchvision torchaudio librosa numpy scikit-learn matplotlib opencv-python pydub
