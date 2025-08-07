import os
import cv2
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Callable

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from sklearn.preprocessing import StandardScaler
from configs.config import Config

class STEMIDataset(Dataset):
    def __init__(self, cine_paths, t2_paths, lge_paths, tabular_paths, labels, transform=None):
        self.cine_paths = cine_paths
        self.t2_paths = t2_paths
        self.lge_paths = lge_paths
        self.tabular_paths = tabular_paths
        self.labels = labels
        self.transform = transform or self.default_transform()
        
    def __len__(self):
        return len(self.labels)
    
    def default_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def load_video(self, path):
        frames = []
        for i in range(25):
            frame_path = f"{path}/cine_frame_{i}.png"
            img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (Config.cine_shape[1], Config.cine_shape[2]))
            img = self.transform(img)
            frames.append(img)
        frames = torch.stack(frames, dim=0)  # (T, C, H, W)
        frames = frames.permute(1, 0, 2, 3)
        return frames
    
    def load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (Config.t2_shape[0], Config.t2_shape[1]))
        img = self.transform(img)
        return img
    
    def load_tabular(self, path):
        tabular_data = pd.read_excel(path).to_numpy()[:, 1:]
        scaler = StandardScaler()
        tabular_data = scaler.fit_transform(tabular_data)
        tabular_data = torch.FloatTensor(tabular_data).squeeze(0)
        return tabular_data#
    
    def __getitem__(self, idx):
        cine = self.load_video(self.cine_paths[idx])  # (T, C, H, W)
        t2 = self.load_image(self.t2_paths[idx])     # (C, H, W)
        lge = self.load_image(self.lge_paths[idx])    # (C, H, W)
        tabular = self.load_tabular(self.tabular_paths[idx]) 
        label = torch.LongTensor([self.labels[idx]])
        
        return cine, t2, lge, tabular, label