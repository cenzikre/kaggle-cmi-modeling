import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from util.data_utils import CMI_Dataset
from typing import Callable, Optional

# Create logger
logger = logging.getLogger(__name__)

# Train model with cross-validation
def train_model_cv(
    model_class: nn.Module,
    dataset: CMI_Dataset,
    model_kwargs: dict = {},
    num_epochs: int = 50, 
    k_folds: int = 4, 
    batch_size: int = 32, 
    lr: float = 1e-3,
    weight_decay: float = 1e-4, 
    eval_every: int = 1, 
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    scheduler_fn: Optional[Callable] = None,
    early_stopping_patience: int = 10,
    early_stopping_delta: float = 2e-2,
):
    labels = dataset.labels
    criterion = nn.CrossEntropyLoss()
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    if scheduler_fn is None:
        scheduler_fn = lambda opt: optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    fold_history = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        logger.info(f"Fold {fold+1}/{k_folds} | Training...")

        train_loader = DataLoader(
            Subset(dataset, train_idx), batch_size=batch_size,
            shuffle=True, collate_fn=dataset.collate_fn
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx), batch_size=batch_size,
            shuffle=False, collate_fn=dataset.collate_fn
        )

        model = model_class(**model_kwargs).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = scheduler_fn(optimizer)
        best_val_loss = float('inf')
        epochs_no_improve = 0

        fold_result = {
            "model": [],
            "train_loss": [],
            "val_loss": [],
            "train_idx": train_idx,
            "val_idx": val_idx,
            "lr": [],
            "best_model": None,
            "model_params": model_kwargs
        }

        for epoch in range(1, num_epochs + 1):
            model.train()
            train_losses = []

            for batch in train_loader:                
                cur_batch_size = batch['labels'].size(0)
                if cur_batch_size <= 1:
                    logger.debug(f"Skipping batch with size {cur_batch_size}")
                    continue

                logits, labels = model.run(batch, device)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())

            # Evaluation
            if epoch % eval_every == 0:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for batch in val_loader:
                        logits, labels = model.run(batch, device)
                        loss = criterion(logits, labels)
                        val_losses.append(loss.item())

                avg_train_loss = np.mean(train_losses)
                avg_val_loss = np.mean(val_losses)
                cur_lr = optimizer.param_groups[0]['lr']
                scheduler.step(avg_val_loss)

                logger.info(f"Fold {fold+1}/{k_folds} | Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

                # Early stopping logic
                if avg_val_loss < best_val_loss - early_stopping_delta:
                    best_val_loss = avg_val_loss
                    best_model = model.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

                # Logging
                fold_result["model"].append(model.state_dict())
                fold_result["train_loss"].append(avg_train_loss)
                fold_result["val_loss"].append(avg_val_loss)
                fold_result["lr"].append(cur_lr)
                fold_result["best_model"] = best_model

        fold_history.append(fold_result)

    return fold_history





# Run as script
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler("training_log.txt"),
            logging.StreamHandler()
        ]
    )

    logger.info("Logger initialized and running as script.")