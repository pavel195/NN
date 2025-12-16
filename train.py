"""
Скрипт для обучения сиамской нейронной сети на датасете LFW.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from model import SiameseNetwork, ContrastiveLoss
from data import LFWDataset, get_transforms

# Фиксируем случайные состояния для воспроизводимости
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Обучение модели на одной эпохе.
    
    Args:
        model: модель сиамской сети
        dataloader: загрузчик данных
        criterion: функция потерь
        optimizer: оптимизатор
        device: устройство (cuda или cpu)
        
    Returns:
        средняя потеря на эпохе
    """
    model.train()
    running_loss = 0.0
    
    for batch_idx, (img1, img2, label) in enumerate(dataloader):
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        
        # Обнуляем градиенты
        optimizer.zero_grad()
        
        # Прямой проход
        output1, output2 = model(img1, img2)
        
        # Вычисление потери
        loss = criterion(output1, output2, label)
        
        # Обратный проход
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')
    
    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """
    Валидация модели.
    
    Args:
        model: модель сиамской сети
        dataloader: загрузчик данных
        criterion: функция потерь
        device: устройство
        
    Returns:
        средняя потеря на валидации
    """
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for img1, img2, label in dataloader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label)
            
            running_loss += loss.item()
    
    return running_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='Обучение сиамской сети для распознавания лиц')
    parser.add_argument('--data_dir', type=str, default='data/lfw/lfw-deepfunneled/lfw-deepfunneled', 
                       help='Путь к директории с датасетом LFW')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Размер батча')
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Количество эпох')
    parser.add_argument('--lr', type=float, default=0.001, 
                       help='Начальный learning rate')
    parser.add_argument('--embedding_dim', type=int, default=128, 
                       help='Размерность эмбеддинга')
    parser.add_argument('--margin', type=float, default=1.0, 
                       help='Маржин для контрастивной потери')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', 
                       help='Директория для сохранения чекпойнтов')
    
    args = parser.parse_args()
    
    # Проверяем наличие GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Используется устройство: {device}')
    
    # Создаем директорию для чекпойнтов
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Разделяем датасет на train/val/test по людям
    print('Загрузка и разделение датасета...')
    persons = [d for d in os.listdir(args.data_dir) 
               if os.path.isdir(os.path.join(args.data_dir, d))]
    
    train_persons, temp_persons = train_test_split(
        persons, test_size=0.3, random_state=42
    )
    val_persons, test_persons = train_test_split(
        temp_persons, test_size=0.5, random_state=42
    )
    
    print(f'Обучающая выборка: {len(train_persons)} человек')
    print(f'Валидационная выборка: {len(val_persons)} человек')
    print(f'Тестовая выборка: {len(test_persons)} человек')
    
    # Создаем датасеты
    train_dataset = LFWDataset(
        args.data_dir, 
        transform=get_transforms('train'),
        mode='train',
        allowed_persons=set(train_persons)
    )
    val_dataset = LFWDataset(
        args.data_dir,
        transform=get_transforms('val'),
        mode='val',
        allowed_persons=set(val_persons)
    )
    
    print(f'Обучающих пар: {len(train_dataset)}')
    print(f'Валидационных пар: {len(val_dataset)}')
    
    # Создаем загрузчики данных
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Создаем модель
    print('Создание модели...')
    model = SiameseNetwork(embedding_dim=args.embedding_dim).to(device)
    
    # Функция потерь и оптимизатор
    criterion = ContrastiveLoss(margin=args.margin)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    
    # История обучения
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print('Начало обучения...')
    for epoch in range(args.epochs):
        print(f'\nЭпоха {epoch + 1}/{args.epochs}')
        print('-' * 50)
        
        # Обучение
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Валидация
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Обновление learning rate
        scheduler.step()
        print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        
        # Сохранение лучшей модели
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, os.path.join(args.checkpoint_dir, 'best_model.pth'))
            print('Модель сохранена!')
    
    # Сохранение финальной модели
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, os.path.join(args.checkpoint_dir, 'final_model.pth'))
    
    # Построение графика потерь
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Эпоха')
    plt.ylabel('Потеря')
    plt.title('Кривые обучения')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.checkpoint_dir, 'training_curves.png'))
    print(f'\nГрафик сохранен: {os.path.join(args.checkpoint_dir, "training_curves.png")}')
    
    print('\nОбучение завершено!')


if __name__ == '__main__':
    main()

