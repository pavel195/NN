"""
Быстрый тест обучения на небольшом подмножестве данных.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

from model import SiameseNetwork, ContrastiveLoss
from data import LFWDataset, get_transforms

# Фиксируем случайные состояния
torch.manual_seed(42)
np.random.seed(42)

def main():
    print("=" * 60)
    print("БЫСТРЫЙ ТЕСТ ОБУЧЕНИЯ")
    print("=" * 60)
    
    data_dir = 'data/lfw/lfw-deepfunneled/lfw-deepfunneled'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}\n")
    
    # Загружаем список людей
    persons = [d for d in os.listdir(data_dir) 
               if os.path.isdir(os.path.join(data_dir, d))]
    
    # Разделяем на train/val/test
    train_persons, temp_persons = train_test_split(
        persons, test_size=0.3, random_state=42
    )
    val_persons, test_persons = train_test_split(
        temp_persons, test_size=0.5, random_state=42
    )
    
    # Берем только первых 50 человек для быстрого теста
    train_persons_subset = set(train_persons[:50])
    val_persons_subset = set(val_persons[:20])
    
    print(f"Обучающих людей: {len(train_persons_subset)}")
    print(f"Валидационных людей: {len(val_persons_subset)}\n")
    
    # Создаем датасеты
    train_dataset = LFWDataset(
        data_dir,
        transform=get_transforms('train'),
        mode='train',
        allowed_persons=train_persons_subset
    )
    val_dataset = LFWDataset(
        data_dir,
        transform=get_transforms('val'),
        mode='val',
        allowed_persons=val_persons_subset
    )
    
    print(f"Обучающих пар: {len(train_dataset)}")
    print(f"Валидационных пар: {len(val_dataset)}\n")
    
    # Создаем загрузчики
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0  # Убираем multiprocessing для стабильности
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )
    
    # Создаем модель
    print("Создание модели...")
    model = SiameseNetwork(embedding_dim=128).to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Начало обучения...\n")
    
    # Обучаем на 1 эпохе
    model.train()
    train_loss = 0.0
    num_batches = min(10, len(train_loader))  # Ограничиваем количество батчей для теста
    
    for batch_idx, (img1, img2, label) in enumerate(train_loader):
        if batch_idx >= num_batches:
            break
            
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        
        optimizer.zero_grad()
        output1, output2 = model(img1, img2)
        loss = criterion(output1, output2, label)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        print(f"Батч {batch_idx + 1}/{num_batches}, Loss: {loss.item():.4f}")
    
    avg_loss = train_loss / num_batches
    print(f"\nСредняя потеря: {avg_loss:.4f}")
    
    # Валидация
    print("\nВалидация...")
    model.eval()
    val_loss = 0.0
    num_val_batches = min(5, len(val_loader))
    
    with torch.no_grad():
        for batch_idx, (img1, img2, label) in enumerate(val_loader):
            if batch_idx >= num_val_batches:
                break
                
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / num_val_batches
    print(f"Средняя валидационная потеря: {avg_val_loss:.4f}")
    
    # Сохраняем модель
    os.makedirs('checkpoint', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'checkpoint/test_model.pth')
    print("\n✓ Модель сохранена в checkpoint/test_model.pth")
    print("\n✓ Тест обучения пройден успешно!")

if __name__ == '__main__':
    main()

