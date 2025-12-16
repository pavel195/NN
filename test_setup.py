"""
Тестовый скрипт для проверки работоспособности кода.
"""

import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from data import LFWDataset, get_transforms
from model import SiameseNetwork, ContrastiveLoss
from torch.utils.data import DataLoader

# Фиксируем случайные состояния
torch.manual_seed(42)
np.random.seed(42)

def test_dataset():
    """Тестирование загрузки датасета."""
    print("=" * 50)
    print("ТЕСТ 1: Загрузка датасета")
    print("=" * 50)
    
    data_dir = 'data/lfw/lfw-deepfunneled/lfw-deepfunneled'
    
    if not os.path.exists(data_dir):
        print(f"ОШИБКА: Директория {data_dir} не найдена!")
        return False
    
    # Получаем список людей
    persons = [d for d in os.listdir(data_dir) 
               if os.path.isdir(os.path.join(data_dir, d))]
    print(f"Найдено людей: {len(persons)}")
    
    if len(persons) < 10:
        print("ОШИБКА: Слишком мало людей в датасете!")
        return False
    
    # Разделяем на train/val/test
    train_persons, temp_persons = train_test_split(
        persons, test_size=0.3, random_state=42
    )
    val_persons, test_persons = train_test_split(
        temp_persons, test_size=0.5, random_state=42
    )
    
    print(f"Train: {len(train_persons)} человек")
    print(f"Val: {len(val_persons)} человек")
    print(f"Test: {len(test_persons)} человек")
    
    # Создаем датасет только для первых 10 человек (для быстрого теста)
    test_persons_subset = set(train_persons[:10])
    
    try:
        dataset = LFWDataset(
            data_dir,
            transform=get_transforms('train'),
            mode='train',
            allowed_persons=test_persons_subset
        )
        print(f"Создан датасет с {len(dataset)} парами")
        
        # Проверяем загрузку одного элемента
        if len(dataset) > 0:
            img1, img2, label = dataset[0]
            print(f"Размер изображения 1: {img1.shape}")
            print(f"Размер изображения 2: {img2.shape}")
            print(f"Метка: {label.item()} ({'разные люди' if label.item() == 1 else 'один человек'})")
            print("✓ Датасет работает корректно!")
            return True
        else:
            print("ОШИБКА: Датасет пуст!")
            return False
    except Exception as e:
        print(f"ОШИБКА при создании датасета: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model():
    """Тестирование модели."""
    print("\n" + "=" * 50)
    print("ТЕСТ 2: Создание и тестирование модели")
    print("=" * 50)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Устройство: {device}")
        
        # Создаем модель
        model = SiameseNetwork(embedding_dim=128).to(device)
        print("✓ Модель создана")
        
        # Тестовый forward pass
        batch_size = 2
        test_input1 = torch.randn(batch_size, 3, 224, 224).to(device)
        test_input2 = torch.randn(batch_size, 3, 224, 224).to(device)
        
        output1, output2 = model(test_input1, test_input2)
        print(f"Размер эмбеддинга 1: {output1.shape}")
        print(f"Размер эмбеддинга 2: {output2.shape}")
        
        # Проверяем нормализацию
        norm1 = torch.norm(output1, dim=1)
        norm2 = torch.norm(output2, dim=1)
        print(f"Норма эмбеддингов (должна быть ~1.0): {norm1.mean().item():.4f}, {norm2.mean().item():.4f}")
        
        # Тестируем функцию потерь
        criterion = ContrastiveLoss(margin=1.0)
        label = torch.tensor([0.0, 1.0]).to(device)  # один человек, разные люди
        loss = criterion(output1, output2, label)
        print(f"Контрастивная потеря: {loss.item():.4f}")
        
        print("✓ Модель работает корректно!")
        return True
    except Exception as e:
        print(f"ОШИБКА при тестировании модели: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader():
    """Тестирование DataLoader."""
    print("\n" + "=" * 50)
    print("ТЕСТ 3: Тестирование DataLoader")
    print("=" * 50)
    
    try:
        data_dir = 'data/lfw/lfw-deepfunneled/lfw-deepfunneled'
        persons = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
        
        train_persons, _ = train_test_split(persons, test_size=0.3, random_state=42)
        test_persons_subset = set(train_persons[:20])  # Берем больше для теста
        
        dataset = LFWDataset(
            data_dir,
            transform=get_transforms('train'),
            mode='train',
            allowed_persons=test_persons_subset
        )
        
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
        
        # Загружаем один батч
        for img1, img2, label in dataloader:
            print(f"Батч - img1: {img1.shape}, img2: {img2.shape}, label: {label.shape}")
            print(f"Метки: {label.tolist()}")
            break
        
        print("✓ DataLoader работает корректно!")
        return True
    except Exception as e:
        print(f"ОШИБКА при тестировании DataLoader: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("Запуск тестов...\n")
    
    results = []
    results.append(("Датасет", test_dataset()))
    results.append(("Модель", test_model()))
    results.append(("DataLoader", test_dataloader()))
    
    print("\n" + "=" * 50)
    print("РЕЗУЛЬТАТЫ ТЕСТОВ")
    print("=" * 50)
    for name, result in results:
        status = "✓ ПРОЙДЕН" if result else "✗ ПРОВАЛЕН"
        print(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n✓ Все тесты пройдены! Код готов к использованию.")
    else:
        print("\n✗ Некоторые тесты провалены. Проверьте ошибки выше.")

