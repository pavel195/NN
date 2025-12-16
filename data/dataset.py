"""
Модуль для загрузки и обработки датасета LFW (Labeled Faces in the Wild).
Создает пары изображений для обучения сиамской сети.
"""

import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split


class LFWDataset(Dataset):
    """
    Датасет для обучения сиамской сети на данных LFW.
    Генерирует пары изображений: положительные (одно лицо) и отрицательные (разные лица).
    """
    
    def __init__(self, data_dir, pairs_file=None, transform=None, mode='train', allowed_persons=None):
        """
        Инициализация датасета.
        
        Args:
            data_dir: путь к директории с изображениями LFW
            pairs_file: путь к файлу с парами (опционально)
            transform: трансформации для изображений
            mode: режим ('train', 'val', 'test')
            allowed_persons: множество разрешенных людей для данного режима (для разделения train/val/test)
        """
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        
        # Загружаем все изображения и их метки
        self.images, self.labels = self._load_images()
        
        # Фильтруем по разрешенным людям, если указано
        if allowed_persons is not None:
            filtered_indices = [i for i, label in enumerate(self.labels) if label in allowed_persons]
            self.images = [self.images[i] for i in filtered_indices]
            self.labels = [self.labels[i] for i in filtered_indices]
        
        # Создаем словарь: метка -> список индексов изображений
        self.label_to_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
        
        # Генерируем пары для обучения
        self.pairs = self._generate_pairs()
        
        # Фиксируем случайное состояние для воспроизводимости
        random.seed(42)
        np.random.seed(42)
    
    def _load_images(self):
        """
        Загружает все изображения из директории LFW.
        
        Returns:
            images: список путей к изображениям
            labels: список меток (имена людей)
        """
        images = []
        labels = []
        
        # Структура LFW: lfw/name/image.jpg
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Директория {self.data_dir} не найдена. "
                           f"Скачайте датасет LFW и распакуйте его.")
        
        for person_name in os.listdir(self.data_dir):
            person_dir = os.path.join(self.data_dir, person_name)
            if not os.path.isdir(person_dir):
                continue
            
            for img_name in os.listdir(person_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(person_dir, img_name)
                    images.append(img_path)
                    labels.append(person_name)
        
        return images, labels
    
    def _generate_pairs(self):
        """
        Генерирует пары изображений для обучения.
        Положительные пары: два изображения одного человека.
        Отрицательные пары: два изображения разных людей.
        
        Returns:
            список кортежей (idx1, idx2, label), где label=0 если один человек, label=1 если разные
        """
        pairs = []
        
        # Положительные пары (один человек)
        for label, indices in self.label_to_indices.items():
            if len(indices) >= 2:
                # Создаем пары из изображений одного человека
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        pairs.append((indices[i], indices[j], 0))
        
        # Отрицательные пары (разные люди)
        # Берем столько же отрицательных пар, сколько положительных
        num_positive = len(pairs)
        labels_list = list(self.label_to_indices.keys())
        
        for _ in range(num_positive):
            # Выбираем двух разных людей
            label1, label2 = random.sample(labels_list, 2)
            idx1 = random.choice(self.label_to_indices[label1])
            idx2 = random.choice(self.label_to_indices[label2])
            pairs.append((idx1, idx2, 1))
        
        # Перемешиваем пары
        random.shuffle(pairs)
        
        return pairs
    
    def __len__(self):
        """Возвращает количество пар в датасете."""
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """
        Возвращает пару изображений и метку.
        
        Args:
            idx: индекс пары
            
        Returns:
            (img1, img2, label): два изображения и метка (0 или 1)
        """
        idx1, idx2, label = self.pairs[idx]
        
        # Загружаем изображения
        img1 = Image.open(self.images[idx1]).convert('RGB')
        img2 = Image.open(self.images[idx2]).convert('RGB')
        
        # Применяем трансформации
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, torch.tensor(label, dtype=torch.float32)


def get_transforms(mode='train'):
    """
    Возвращает трансформации для изображений.
    
    Args:
        mode: режим ('train', 'val', 'test')
        
    Returns:
        объект transforms.Compose
    """
    if mode == 'train':
        # Для обучения добавляем аугментации
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet нормализация
        ])
    else:
        # Для валидации и теста только базовые трансформации
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def split_dataset(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Разбивает датасет на train/val/test так, чтобы изображения одного человека
    не попадали в разные множества.
    
    Args:
        data_dir: путь к директории с данными LFW
        train_ratio: доля обучающей выборки
        val_ratio: доля валидационной выборки
        test_ratio: доля тестовой выборки
        
    Returns:
        словарь с путями к директориям для каждого множества
    """
    # Загружаем всех людей
    persons = [d for d in os.listdir(data_dir) 
               if os.path.isdir(os.path.join(data_dir, d))]
    
    # Разбиваем людей (не изображения!) на train/val/test
    train_persons, temp_persons = train_test_split(
        persons, test_size=(1 - train_ratio), random_state=42
    )
    val_persons, test_persons = train_test_split(
        temp_persons, test_size=test_ratio / (val_ratio + test_ratio), random_state=42
    )
    
    return {
        'train': (data_dir, train_persons),
        'val': (data_dir, val_persons),
        'test': (data_dir, test_persons)
    }

