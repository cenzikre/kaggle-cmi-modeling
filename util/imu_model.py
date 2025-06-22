import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from util.data_utils import CMI_Dataset


class IMU_LSTM_Model(nn.Module):
    def __init__(self, num_classes=18, input_dim_demo=7, input_dim_imu=7, hidden_dim=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim_imu, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + input_dim_demo, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, num_classes)
        )

    def forward(self, demo, imu, mask=None):
        # imu: [B, T, D]
        lstm_out, _ = self.lstm(imu)  # [B, T, 2H]

        if mask is not None:
            mask = mask.unsqueeze(-1)  # [B, T, 1]
            lstm_out = lstm_out * mask  # zero out padding
            summed = lstm_out.sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1e-6)
            pooled = summed / lengths  # mean over valid steps
        else:
            pooled = lstm_out.mean(dim=1)  # [B, 2H]

        x = torch.cat([pooled, demo], dim=1)  # [B, 2H + demo]
        return self.mlp(self.dropout(x))
    
    def run(self, batch, device=None):
        if device is None:
            device = next(self.parameters()).device

        labels = batch["labels"].to(device)
        demo = batch["demo"].to(device)
        seq_imu = batch["seq_imu"].to(device)
        mask = batch["mask"].to(device)
        logits = self.forward(demo, seq_imu, mask)

        return logits, labels
    
    def predict(self, dataset, pred_idx, batch_size=32, device=None):
        if device is None:
            device = next(self.parameters()).device

        pred_loader = DataLoader(
            Subset(dataset, pred_idx), batch_size=batch_size,
            shuffle=False, collate_fn=CMI_Dataset.collate_fn
        )

        labels, preds = [], []
        self.eval()

        with torch.no_grad():
            for batch in pred_loader:
                label = batch["labels"].to(device)
                demo = batch["demo"].to(device)
                seq_imu = batch["seq_imu"].to(device)
                mask = batch["mask"].to(device)

                logits = self(demo, seq_imu, mask)
                pred = logits.argmax(dim=1)
                labels.append(label)
                preds.append(pred)

        return torch.cat(labels, dim=0).cpu(), torch.cat(preds, dim=0).cpu()