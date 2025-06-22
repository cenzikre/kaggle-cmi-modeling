import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from util.data_utils import CMI_Dataset


class TOF_Model(nn.Module):
    def __init__(
        self, 
        num_classes=18, 
        input_dim_demo=7, 
        input_dim_1d=12, 
        hidden_dim=32, 
        num_layers=1
    ):
        super().__init__()
