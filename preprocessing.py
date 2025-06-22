import numpy as np
import pandas as pd
import torch, pickle
from pathlib import Path
from util.data_utils import CMI_Dataset, MaskedStandardScaler, label_map, good_labels


project_root = Path(__file__).parent
raw_data_dir = project_root / "data" / "raw" / "cmi-detect-behavior-with-sensor-data"

# Load datasets
train = pd.read_csv(raw_data_dir / "train.csv")
train_demographics = pd.read_csv(raw_data_dir / "train_demographics.csv")
test = pd.read_csv(raw_data_dir / "test.csv")
test_demographics = pd.read_csv(raw_data_dir / "test_demographics.csv")

# Column names
acc_cols = [col for col in train.columns if "acc" in col]
rot_cols = [col for col in train.columns if "rot" in col]
thm_cols = [col for col in train.columns if "thm" in col]
tof1_cols = [col for col in train.columns if col.startswith("tof_1")]
tof2_cols = [col for col in train.columns if col.startswith("tof_2")]
tof3_cols = [col for col in train.columns if col.startswith("tof_3")]
tof4_cols = [col for col in train.columns if col.startswith("tof_4")]
tof5_cols = [col for col in train.columns if col.startswith("tof_5")]
tof_cols = tof1_cols + tof2_cols + tof3_cols + tof4_cols + tof5_cols
itof1_cols = [f"i{col}" for col in tof1_cols]
itof2_cols = [f"i{col}" for col in tof2_cols]
itof3_cols = [f"i{col}" for col in tof3_cols]
itof4_cols = [f"i{col}" for col in tof4_cols]
itof5_cols = [f"i{col}" for col in tof5_cols]
itof_cols = itof1_cols + itof2_cols + itof3_cols + itof4_cols + itof5_cols

# Create tof mask
train.loc[:, itof_cols] = ((train[tof_cols]!=-1) & (train[tof_cols].notna())).astype(float).values
test.loc[:, itof_cols] = ((test[tof_cols]!=-1) & (test[tof_cols].notna())).astype(float).values

# Standardize
unmasked_cols = acc_cols + rot_cols + thm_cols
masked_cols = tof1_cols + tof2_cols + tof3_cols + tof4_cols + tof5_cols
X_cols = unmasked_cols + masked_cols
mask_cols = [None] * len(unmasked_cols) + [f"i{col}" for col in masked_cols]

scaler = MaskedStandardScaler()
scaler.fit_transform(train, X_cols, mask_cols, inplace=True)
scaler.transform(test, inplace=True)

# Create gesture id
train['gesture_id'] = train['gesture'].map(label_map)
# test['gesture_id'] = test['gesture'].map(label_map)

train['gesture_id2'] = np.where(train['gesture_id'].isin(good_labels), 0, 1)
# test['gesture_id2'] = np.where(test['gesture_id'].isin(good_labels), 0, 1)

# Create feature tensors
cols = ['sequence_id', 'sequence_counter'] + acc_cols + rot_cols + thm_cols
feature_imu = acc_cols + rot_cols
feature_1d = acc_cols + rot_cols + thm_cols
feature_2d_tof = tof1_cols + tof2_cols + tof3_cols + tof4_cols + tof5_cols
feature_2d_itof = itof1_cols + itof2_cols + itof3_cols + itof4_cols + itof5_cols

def create_sequence_tensors(df):
    feature_imu_sequence = []
    feature_1d_sequence = []
    feature_2d_tof_sequence = []
    feature_2d_itof_sequence = []
    labels = []
    labels2 = []

    for _, seq_df in df.groupby("sequence_id"):
        feature_imu_sequence.append(torch.tensor(seq_df[feature_imu].values))
        feature_1d_sequence.append(torch.tensor(seq_df[feature_1d].values))

        flat_tensor_tof = torch.tensor(seq_df[feature_2d_tof].values)
        feature_2d_tof_sequence.append(flat_tensor_tof.reshape(-1, 5, 8, 8))

        flat_tensor_itof = torch.tensor(seq_df[feature_2d_itof].values)
        feature_2d_itof_sequence.append(flat_tensor_itof.reshape(-1, 5, 8, 8))

        if 'gesture_id' in seq_df.columns:
            labels.append(seq_df['gesture_id'].iloc[-1].item())
            labels2.append(seq_df['gesture_id2'].iloc[-1].item())

    labels = torch.tensor(labels)
    labels2 = torch.tensor(labels2)
    return feature_imu_sequence, feature_1d_sequence, feature_2d_tof_sequence, feature_2d_itof_sequence, labels, labels2

train_feature_imu_sequence, train_feature_1d_sequence, train_feature_2d_tof_sequence, train_feature_2d_itof_sequence, train_labels, train_labels2 = create_sequence_tensors(train)
test_feature_imu_sequence, test_feature_1d_sequence, test_feature_2d_tof_sequence, test_feature_2d_itof_sequence, test_labels, test_labels2 = create_sequence_tensors(test)

# Create demo tensors
train_seq_demographics = pd.merge(train[['sequence_id', 'subject']].drop_duplicates(), train_demographics, on='subject', how='left')
test_seq_demographics = pd.merge(test[['sequence_id', 'subject']].drop_duplicates(), test_demographics, on='subject', how='left')

demo_features = ['adult_child', 'age', 'sex', 'handedness', 'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm']
demo_scaler = MaskedStandardScaler()
demo_scaler.fit_transform(train_seq_demographics, demo_features, inplace=True)
demo_scaler.transform(test_seq_demographics, inplace=True)

train_feature_demo = torch.tensor(train_seq_demographics[demo_features].values)
test_feature_demo = torch.tensor(test_seq_demographics[demo_features].values)

# Create dataset
train_dataset = CMI_Dataset(train_labels, train_labels2, train_feature_demo, train_feature_imu_sequence, train_feature_1d_sequence, train_feature_2d_tof_sequence, train_feature_2d_itof_sequence)
test_dataset = CMI_Dataset(test_labels, test_labels2, test_feature_demo, test_feature_imu_sequence, test_feature_1d_sequence, test_feature_2d_tof_sequence, test_feature_2d_itof_sequence)

# Save datasets
with open(project_root / "data" / "processed" / "train_dataset_v1.pkl", "wb") as f:
    pickle.dump(train_dataset, f)

with open(project_root / "data" / "processed" / "test_dataset_v1.pkl", "wb") as f:
    pickle.dump(test_dataset, f)