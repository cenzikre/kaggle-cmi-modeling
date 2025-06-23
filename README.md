# Kaggle CMI Behavior Detection Modeling

Detect body-focused repetitive behaviors (BFRBs) from wearable sensor data using a multimodal neural network—built for the [CMI Detect Behavior](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data) competition.

## Project Overview
- **Objective**: Identify BFRBs and specific gesture types using synchronized sensor streams: Time-of-Flight (TOF), Infrared TOF masks, IMU (accelerometer/gyroscope), and thermal data.
- **Approach**: Designed a hybrid CNN+BiLSTM model with per-frame TOF image feature extraction (including mask-aware CNN fusion), temporal feature aggregation, and structured sensor/channel attention pooling.
- Preprocessing includes sequence padding, masked inputs for variable sequence lengths, stratified k-fold cross-validation, learning-rate scheduling, and early stopping to handle a limited dataset (~8K labeled samples).

## Results
- **BFRB vs. non-BFRB**: ~98% accuracy on binary classification.
- **Fine-grained gesture detection**: ~58% accuracy across 18 gesture classes.

---

## Repo Contents

```
/
├─ data/                # (Not included) sensor .pt or .np files
├─ util/
|   ├─ data_utils.py    # Data preprocessing helpers
│   ├─ imu_model.py     # IMU-only model (LSTM)
│   ├─ tof_model.py     # TOF_CNN + BiLSTM model implementation
│   └─ train_utils.py   # Training loops, cross-validation, logging
├─ notebooks.ipynb      # Data exploration, model training & evaluation
└─ preprocessing.py     # Preprocessing script
```

---

## Architecture Details

- **TOF Extractor**:  
  CNN with mask awareness (`Conv2d(2→8→16)` + BatchNorm + ReLU + AdaptiveAvgPool)  
  → Device-level pooling (mean / max / attention) → Dimensionality reduction.

- **Temporal Fusion**:  
  Concatenate TOF features `[B, T, D]` with IMU features `[B, T, 12]` → BiLSTM (hidden = 32, bidirectional)  
  → Mean pooling → Final MLP classifiers (binary & 18-class).

- **Training Tactics**:  
  Stratified K-fold, LR scheduler, early stopping, batch-size guard, masked padding, gradient clipping.

---

## Key Engineering Highlights

- **Mask + Image Channel Fusion**: TOF + mask as dual-channel input ensures CNN learns both sensor values and validity.
- **Cross-modal feature fusion**: Early concatenation of per-frame TOF and per-timestep IMU features.
- **Device-level attention pooling**: Learned weights across multiple TOF devices—enhancing per-timestep feature importance.

---

## Data Format

- `tof`: `[B, T, 5, 8, 8]` – 5-device TOF values per timestep  
- `itof`: same shape, binary mask validity  
- `seq_1d`: `[B, T, 12]` – IMU + thermal features  
- `demo`: `[B, 7]` – Subject-level demographic inputs  
- `mask`: `[B, T]` – Boolean mask for valid time steps  

---

## Future Directions

- Optimize IMU-only model with attention based on pooling  
- Experimenting with transformer encoders across time and addition methods to boost fine-grained classification

---

## Repository

Code & model artifacts: [github.com/cenzikre/kaggle-cmi-modeling](https://github.com/cenzikre/kaggle-cmi-modeling)

---

## 📝 License

This project is open-source under the MIT License. Feel free to explore and reuse!

