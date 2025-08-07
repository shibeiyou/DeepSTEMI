import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import cv2
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import Counter
from utils.dataset import STEMIDataset
from utils.DeepSTEMI import DeepSTEMI
from utils.lossfunc import WeightedCrossEntropyLoss
from configs.config import Config

def evaluate_model(model, data_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for cine, t2, lge, tabular, labels in data_loader:
            cine = cine.to(Config.device)
            t2 = t2.to(Config.device)
            lge = lge.to(Config.device)
            tabular = tabular.to(Config.device)
            labels = labels.to(Config.device)
            
            outputs = model(cine, t2, lge, tabular)
            loss = criterion(outputs, labels.squeeze(1))
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.squeeze()).sum().item()
    
    loss = running_loss / len(data_loader)
    acc = 100 * correct / total
    return loss, acc

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for cine, t2, lge, tabular, labels in train_loader:
            cine = cine.to(Config.device)
            t2 = t2.to(Config.device)
            lge = lge.to(Config.device)
            tabular = tabular.to(Config.device)
            labels = labels.to(Config.device)
            outputs = model(cine, t2, lge, tabular)
            loss = criterion(outputs, labels.squeeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.squeeze()).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%')
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'MODELS/DeepSTEMI_best.pth')
    
    print(f'Training complete. Best Val Acc: {best_acc:.2f}%')

def main():
    data_dir='dataset'
    cine_paths = [os.path.join(data_dir, 'cine', i) for i in os.listdir(os.path.join(data_dir, 'cine'))]  
    t2_paths = [os.path.join(data_dir, 't2', i) for i in os.listdir(os.path.join(data_dir, 't2'))]
    lge_paths = [os.path.join(data_dir, 'lge', i) for i in os.listdir(os.path.join(data_dir, 'lge'))]
    tabular_paths = [os.path.join(data_dir, 'tab', i) for i in os.listdir(os.path.join(data_dir, 'tab'))]
    labels = [int(i.split('_')[0]) for i in os.listdir(os.path.join(data_dir, 'cine'))]
    labels = np.array(labels)
    
    class_counts = Counter(labels)
    weights = torch.tensor([
        1.0 / class_counts[0],  
        1.0 / class_counts[1]   
    ], dtype=torch.float32)
    
    train_idx, val_idx = train_test_split(
        np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=42
    )
    train_dataset = STEMIDataset(
        [cine_paths[i] for i in train_idx],
        [t2_paths[i] for i in train_idx],
        [lge_paths[i] for i in train_idx],
        [tabular_paths[i] for i in train_idx],
        labels[train_idx]
    )
    
    val_dataset = STEMIDataset(
        [cine_paths[i] for i in val_idx],
        [t2_paths[i] for i in val_idx],
        [lge_paths[i] for i in val_idx],
        [tabular_paths[i] for i in val_idx],
        labels[val_idx]
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=Config.batch_size,
        shuffle=True, num_workers=Config.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=Config.batch_size,
        shuffle=False, num_workers=Config.num_workers
    )
    
    model = DeepSTEMI().to(Config.device)
    
    criterion = WeightedCrossEntropyLoss(weights=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.learning_rate,
        weight_decay=Config.weight_decay
    )
    
    train_model(
        model, train_loader, val_loader,
        criterion, optimizer, Config.epochs
    )

if __name__ == '__main__':
    main()


