"""
Сиамская нейронная сеть для распознавания лиц.
Архитектура основана на ResNet-50 с двумя идентичными ветвями.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class SiameseNetwork(nn.Module):
    """
    Сиамская сеть для верификации лиц.
    
    Состоит из двух идентичных ветвей на основе ResNet-50,
    которые извлекают 128-мерные эмбеддинги из изображений лиц.
    """
    
    def __init__(self, embedding_dim=128):
        """
        Инициализация сиамской сети.
        
        Args:
            embedding_dim: размерность выходного эмбеддинга (по умолчанию 128)
        """
        super(SiameseNetwork, self).__init__()
        
        # Загружаем предобученную ResNet-50
        resnet = models.resnet50(pretrained=True)
        
        # Удаляем последний fully connected слой
        # Оставляем все слои до avgpool
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Добавляем новый fully connected слой для получения эмбеддинга
        # ResNet-50 после avgpool дает 2048 признаков
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, embedding_dim)
        )
        
        # Инициализация весов нового слоя
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward_one(self, x):
        """
        Прямой проход для одного изображения.
        
        Args:
            x: входное изображение (batch_size, 3, 224, 224)
            
        Returns:
            эмбеддинг изображения (batch_size, embedding_dim)
        """
        # Извлечение признаков через ResNet
        x = self.backbone(x)
        # Flatten
        x = x.view(x.size(0), -1)
        # Получение эмбеддинга
        x = self.fc(x)
        # L2 нормализация для стабильности обучения
        x = nn.functional.normalize(x, p=2, dim=1)
        return x
    
    def forward(self, input1, input2):
        """
        Прямой проход для пары изображений.
        
        Args:
            input1: первое изображение (batch_size, 3, 224, 224)
            input2: второе изображение (batch_size, 3, 224, 224)
            
        Returns:
            эмбеддинги для обоих изображений
        """
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2


class ContrastiveLoss(nn.Module):
    """
    Контрастивная функция потерь для обучения сиамской сети.
    
    Формула: L = (1-Y) * 0.5 * D^2 + Y * 0.5 * max(0, margin - D)^2
    где Y = 1 если разные люди, Y = 0 если один человек
    """
    
    def __init__(self, margin=1.0):
        """
        Инициализация контрастивной потери.
        
        Args:
            margin: маржин для разных лиц (по умолчанию 1.0)
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        """
        Вычисление контрастивной потери.
        
        Args:
            output1: эмбеддинг первого изображения
            output2: эмбеддинг второго изображения
            label: метка (1 если разные люди, 0 если один человек)
            
        Returns:
            значение потери
        """
        # Евклидово расстояние между эмбеддингами
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        
        # Контрастивная потеря
        # Если label=0 (один человек), штрафуем за большое расстояние
        # Если label=1 (разные люди), штрафуем за маленькое расстояние (меньше маржина)
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        
        return loss_contrastive

