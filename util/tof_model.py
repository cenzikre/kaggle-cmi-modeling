import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from util.data_utils import CMI_Dataset
from typing import Literal


class TOF_Model(nn.Module):
    def __init__(
        self, 
        num_classes=18, 
        input_dim_demo=7, 
        input_dim_1d=12, 
        hidden_dim=32, 
        num_lstm_layers=1,
        num_2d_features=16,
        dropout_rate=0.2,
        pooling_type: Literal['max', 'avg', 'attn', 'mlp'] = 'avg'
    ):
        super().__init__()
        self.pooling_type = pooling_type
        
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=num_2d_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_2d_features),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.device_attn = nn.Sequential(
            nn.Linear(num_2d_features, 1),
        )

        self.device_linear = nn.Sequential(
            nn.Linear(num_2d_features * 5, num_2d_features),
            nn.LayerNorm(num_2d_features),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.lstm = nn.LSTM(
            input_size=input_dim_1d+num_2d_features,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + input_dim_demo, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, num_classes)
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.lstm_bn = nn.BatchNorm1d(hidden_dim * 2)

    def seq_mask_out(self, x, mask):
        if mask.dtype == torch.bool:
            mask = mask.float()
        
        while mask.dim() < x.dim():
            mask = mask.unsqueeze(-1)
        
        return x * mask
    
    def device_pooling(self, x: torch.Tensor, pooling_type: str):
        if pooling_type == 'max':
            return torch.max(x, dim=2)[0]
        
        elif pooling_type == 'avg':
            return torch.mean(x, dim=2)
        
        elif pooling_type == 'attn':
            B, T, C, D = x.size()
            x_flat = x.view(B*T*C, D)
            attn_scores = self.device_attn(x_flat)
            attn_scores = attn_scores.view(B, T, C)
            attn_weights = torch.softmax(attn_scores, dim=2)
            attn_weights = attn_weights.unsqueeze(-1)
            return torch.sum(x * attn_weights, dim=2)
        
        elif pooling_type == 'mlp':
            B, T, C, D = x.size()
            x = x.view(B, T, C*D)
            x = self.device_linear(x)
            return x
        
        else:
            raise ValueError(f"Invalid pooling type: {pooling_type}")

    def tof_extractor(self, tof, itof, mask, pooling_type: str):
        tof = self.seq_mask_out(tof, mask)
        itof = self.seq_mask_out(itof, mask)
        
        # stack tof and itof
        B, T, C, H, W = tof.size() # [B, T, 5, 8, 8]
        x = torch.stack([tof, itof], dim=3) # [B, T, 5, 2, 8, 8]
        x = x.view(B*T*C, 2, H, W) # [B*T*C, 2, 8, 8]

        # extract features
        x = self.cnn(x) # [B*T*C, D, 1, 1]
        x = x.view(B, T, C, -1) # [B, T, C, D]
        x = self.seq_mask_out(x, mask) # [B, T, C, D]
        x = self.device_pooling(x, pooling_type=pooling_type) # [B, T, D]

        return x
    
    def forward(self, demo, seq_1d, tof, itof, mask):
        # Extract TOF features
        x_tof = self.tof_extractor(tof, itof, mask, pooling_type=self.pooling_type) # [B, T, D_2d]

        # Extract All Features
        x_seq = torch.cat([x_tof, seq_1d], dim=2) # [B, T, D_2d + D_1d]
        x_lstm, _ = self.lstm(x_seq) # [B, T, 2H]
        x_lstm_masked = self.seq_mask_out(x_lstm, mask) # [B, T, 2H]

        # Pooling sensor features over time
        summed = x_lstm_masked.sum(dim=1)
        lengths = mask.unsqueeze(-1).sum(dim=1).clamp(min=1e-6)
        x_pooled = summed / lengths # [B, 2H]
        x_pooled = self.lstm_bn(x_pooled)
        x_pooled = self.dropout(x_pooled)

        # Fuse sensor features with demo features and predict
        x = torch.cat([x_pooled, demo], dim=1) # [B, 2H + D_demo]
        return self.mlp(self.dropout(x)) # [B, num_classes]
    
    def get_data(self, batch):
        device = next(self.parameters()).device
        labels = batch["labels"].to(device)
        demo = batch["demo"].float().to(device)
        seq_1d = batch["seq_1d"].float().to(device)
        tof = batch["seq_2d_tof"].float().to(device)
        itof = batch["seq_2d_itof"].float().to(device)
        mask = batch["mask"].to(device)
        return {"labels": labels, "demo": demo, "seq_1d": seq_1d, "tof": tof, "itof": itof, "mask": mask}
    
    def run(self, batch, device=None):
        data = self.get_data(batch)
        labels = data.pop("labels")
        logits = self.forward(**data)
        return logits, labels
    
    def predict(self, dataset, pred_idx, batch_size=32):
        pred_loader = DataLoader(
            Subset(dataset, pred_idx), batch_size=batch_size,
            shuffle=False, collate_fn=CMI_Dataset.collate_fn
        )

        labels, preds = [], []
        self.eval()
        
        with torch.no_grad():
            for batch in pred_loader:
                logits, label = self.run(batch)
                preds.append(logits.argmax(dim=1))
                labels.append(label)

        return torch.cat(labels, dim=0).cpu(), torch.cat(preds, dim=0).cpu()
                
