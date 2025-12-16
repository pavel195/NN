"""
Скрипт для оценки обученной сиамской сети на тестовом множестве.
Вычисляет метрики Accuracy, Precision, Recall, F1-score.
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from model import SiameseNetwork
from data import LFWDataset, get_transforms


def evaluate(model, dataloader, device, threshold=0.5):
    """
    Оценка модели на тестовом множестве.
    
    Args:
        model: обученная модель сиамской сети
        dataloader: загрузчик тестовых данных
        device: устройство (cuda или cpu)
        threshold: порог для классификации (расстояние меньше порога = один человек)
        
    Returns:
        словарь с метриками
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for img1, img2, label in dataloader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            
            # Получаем эмбеддинги
            output1, output2 = model(img1, img2)
            
            # Вычисляем евклидово расстояние
            euclidean_distance = nn.functional.pairwise_distance(output1, output2)
            
            # Предсказание: если расстояние меньше порога, то один человек (0), иначе разные (1)
            # Но у нас label=0 для одного человека, label=1 для разных
            # Поэтому: если расстояние < threshold, предсказываем 0 (один человек)
            predictions = (euclidean_distance > threshold).float()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    
    # Вычисляем метрики
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': all_predictions,
        'labels': all_labels
    }


def find_best_threshold(model, dataloader, device):
    """
    Находит оптимальный порог для классификации на валидационном множестве.
    
    Args:
        model: обученная модель
        dataloader: загрузчик валидационных данных
        device: устройство
        
    Returns:
        оптимальный порог
    """
    model.eval()
    all_distances = []
    all_labels = []
    
    with torch.no_grad():
        for img1, img2, label in dataloader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            
            output1, output2 = model(img1, img2)
            euclidean_distance = nn.functional.pairwise_distance(output1, output2)
            
            all_distances.extend(euclidean_distance.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    
    # Перебираем пороги и находим лучший по F1-score
    best_threshold = 0.5
    best_f1 = 0
    
    thresholds = np.arange(0.1, 2.0, 0.05)
    for threshold in thresholds:
        predictions = (np.array(all_distances) > threshold).astype(float)
        f1 = f1_score(all_labels, predictions, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold


def main():
    parser = argparse.ArgumentParser(description='Оценка сиамской сети для распознавания лиц')
    parser.add_argument('--data_dir', type=str, default='data/lfw/lfw-deepfunneled/lfw-deepfunneled', 
                       help='Путь к директории с датасетом LFW')
    parser.add_argument('--checkpoint', type=str, default='checkpoint/best_model.pth', 
                       help='Путь к чекпойнту модели')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Размер батча')
    parser.add_argument('--embedding_dim', type=int, default=128, 
                       help='Размерность эмбеддинга')
    parser.add_argument('--threshold', type=float, default=None, 
                       help='Порог для классификации (если None, будет найден оптимальный)')
    
    args = parser.parse_args()
    
    # Проверяем наличие GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Используется устройство: {device}')
    
    # Загружаем модель
    print('Загрузка модели...')
    model = SiameseNetwork(embedding_dim=args.embedding_dim).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Модель загружена из эпохи {checkpoint.get("epoch", "unknown")}')
    
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
    
    # Создаем тестовый датасет
    test_dataset = LFWDataset(
        args.data_dir,
        transform=get_transforms('test'),
        mode='test',
        allowed_persons=set(test_persons)
    )
    
    print(f'Тестовых пар: {len(test_dataset)}')
    
    # Создаем загрузчик данных
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Если порог не указан, находим оптимальный на валидации
    if args.threshold is None:
        print('Поиск оптимального порога на валидационном множестве...')
        val_dataset = LFWDataset(
            args.data_dir,
            transform=get_transforms('val'),
            mode='val',
            allowed_persons=set(val_persons)
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        threshold = find_best_threshold(model, val_loader, device)
        print(f'Найден оптимальный порог: {threshold:.4f}')
    else:
        threshold = args.threshold
        print(f'Используется порог: {threshold:.4f}')
    
    # Оценка на тестовом множестве
    print('\nОценка на тестовом множестве...')
    results = evaluate(model, test_loader, device, threshold=threshold)
    
    # Вывод результатов
    print('\n' + '=' * 50)
    print('РЕЗУЛЬТАТЫ НА ТЕСТОВОМ МНОЖЕСТВЕ')
    print('=' * 50)
    print(f'Accuracy:  {results["accuracy"]:.4f}')
    print(f'Precision: {results["precision"]:.4f}')
    print(f'Recall:    {results["recall"]:.4f}')
    print(f'F1-score: {results["f1_score"]:.4f}')
    print('=' * 50)
    
    # Сохранение результатов
    results_file = os.path.join(os.path.dirname(args.checkpoint), 'test_results.txt')
    with open(results_file, 'w') as f:
        f.write('Результаты на тестовом множестве\n')
        f.write('=' * 50 + '\n')
        f.write(f'Accuracy:  {results["accuracy"]:.4f}\n')
        f.write(f'Precision: {results["precision"]:.4f}\n')
        f.write(f'Recall:    {results["recall"]:.4f}\n')
        f.write(f'F1-score: {results["f1_score"]:.4f}\n')
        f.write(f'Threshold: {threshold:.4f}\n')
    
    print(f'\nРезультаты сохранены в {results_file}')


if __name__ == '__main__':
    main()

