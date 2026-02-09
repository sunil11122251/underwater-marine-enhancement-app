# 🌊 Adaptive Image Enhancement Techniques for Underwater Marine Visuals
> **A Hybrid Parallel Enhancement Framework using Classical Processing and ADGOL Deep Refinement**

This project presents a deployable web-based system for restoring degraded underwater images affected by color cast, scattering, low contrast, and poor visibility. The approach follows a **parallel enhancement architecture** where LAB-based contrast enhancement and visibility restoration operate simultaneously and are later fused, followed by deep learning refinement using the ADGOL network.

---

## 📌 Project Overview

Underwater Marine images typically suffer from:

- Wavelength-dependent absorption of red light  
- Scattering from suspended particles  
- Blue/green color dominance  
- Loss of contrast and structural details  

The proposed system addresses these issues through a **hybrid adaptive framework** combining classical image processing with a lightweight CNN.

---

## 🎯 Objectives

- Restore natural underwater color balance  
- Improve visibility and contrast without artifacts  
- Preserve structural and edge information  
- Provide stage-wise analysis and metrics  
- Deliver a user-friendly enhancement tool

---

## 🏗 System Architecture (Parallel Model)

```
                        ```
              Start
                │
                ▼
       Captured Underwater Image
                │
                ▼
      Dynamic Parameter Estimation
                │
                ▼
        Adaptive White Balancing
                │
                ▼ 
         Color Correction
                 │
  ┌──────────────────────────────────────────┐
  ▼                                          ▼
Contrast Enhancement               Visibility Restoration
  │                                          │
  └───────────────┬──────────────────────────┘
                  ▼
          Adaptive Fusion
     (Weighted Linear Combination)
                  │
                  ▼
         ADGOL Refinement
                  │
                  ▼
         Final Enhanced Image

---

## 🧪 Technical Methodology

### 1. Pre-Processing
- **White Balance Correction**  
  Removes dominant color cast and normalizes RGB channels.

### 2. Parallel Enhancement Branches

#### Branch A – LAB Correction
- CLAHE on luminance channel  
- Local contrast improvement  
- Structural detail preservation

#### Branch B – Visibility Restoration
- Adaptive gamma adjustment  
- Illumination normalization  
- Dehazing effect

### 3. Fusion Strategy
- Weighted fusion of both branches  
- Complementary feature integration  
- Balanced color and detail

### 4. Deep Refinement – ADGOL
- Lightweight CNN model  
- Residual artifact removal  
- Edge & texture enhancement

---

## 📊 Evaluation Metrics

The system provides quantitative and perceptual evaluation:

- **PSNR** – Peak Signal-to-Noise Ratio  
- **SSIM** – Structural Similarity  
- **UCIQE** – Underwater Color Image Quality Evaluation  
- **UIQM** – Underwater Image Quality Measure  
- Stage-wise perceptual analysis (UCIQE/UIQM)

---

## 🖥 Application Features

✔ Upload any underwater image  
✔ View all processing stages sequentially  
✔ Before/After comparison  
✔ Metrics on demand  
✔ Stage-wise quality table  
✔ Download:
- Individual stage images  
- All results as ZIP

✔ Professional multi-tab dashboard:
- Enhancement  
- Stage Visualization  
- Evaluation  
- Download Center

---

## 📁 Project Structure

```
underwater-marine-enhancement-app/
│
├── app.py            # Streamlit dashboard
├── model.py          # ADGOL architecture
├── utils.py          # Parallel pipeline
├── metrics.py        # Evaluation logic
├── requirements.txt  # Dependencies
├── weights/
│   └── adgol.pth     # Trained model
├── README.md
└── MIT License
```

---

## ⚙ Installation & Usage

### 1. Clone Repository

```bash
git clone https://github.com/sunil11122251/underwater-marine-enhancement-app.git
cd underwater-marine-enhancement-app
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Application

```bash
streamlit run app.py
```

---

## 🖼 Workflow

1. Upload underwater image  
2. White balance preprocessing  
3. **Parallel processing**
   - LAB correction branch  
   - Visibility restoration branch  
4. Fusion of both results  
5. ADGOL deep refinement  
6. Metric generation & download

---

## 🔍 Stage Description

| Stage | Type | Purpose |
|-----|------|---------|
| White Balance | Preprocess | Color cast removal |
| LAB Correction | Parallel A | Contrast enhancement |
| Visibility | Parallel B | Illumination fix |
| Fusion | Integration | Combine strengths |
| ADGOL | Deep Model | Final quality boost |

---

## 🧰 Tools & Technologies

- **Frontend**: Streamlit  
- **Deep Learning**: PyTorch  
- **Image Processing**: OpenCV  
- **Metrics**: Scikit-Image  
- **Language**: Python

---

## 🔮 Future Scope

- Real-time video enhancement  
- GAN-based refinement  
- True no-reference IQA  
- Mobile deployment  
- Batch mode processing

---

## 👨‍🎓 Academic Context

- Domain: Underwater Image Processing  
- Approach: Parallel Hybrid Framework  
- Output: Deployable Application  
- Type: B.Tech Project

---

## 👤 Author

**Sunil**  
**Syam Kiran** 
**Pavan** 
**Bhanusree** 
B.Tech CSE Students  
Project: Adaptive Image Enhancement Techniques for Underwater Marine Visuals

---

## 📜 License

Developed for educational and research purposes.

---

## 🙏 Acknowledgement

This work is inspired by recent research in underwater image restoration using hybrid classical–deep learning paradigms.

---

### ⭐ If you find this useful, please star the repository!
