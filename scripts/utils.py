import os
import random
from functools import partial
import numpy as np
import timm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torchmetrics
from transformers import AutoModel, AutoTokenizer
from scripts.dataset import MultimodalDataset, collate_fn, get_transforms


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def set_requires_grad(module: nn.Module, unfreeze_pattern="", verbose=False):
    if len(unfreeze_pattern) == 0:
        for _, param in module.named_parameters():
            param.requires_grad = False
        return

    pattern = unfreeze_pattern.split("|")

    for name, param in module.named_parameters():
        if any([name.startswith(p) for p in pattern]):
            param.requires_grad = True
            if verbose:
                print(f"Разморожен слой: {name}")
        else:
            param.requires_grad = False


class MultimodalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        self.image_model = timm.create_model(
            config.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0
        )

        self.text_proj = nn.Linear(self.text_model.config.hidden_size, config.HIDDEN_DIM)
        self.image_proj = nn.Linear(self.image_model.num_features, config.HIDDEN_DIM)

        # Проекция для числовых признаков mass и n_ingredients
        self.numeric_proj = nn.Linear(2, config.HIDDEN_DIM // 4)  # уменьшаем размерность

        # Классификатор теперь принимает объединённые признаки
        combined_dim = config.HIDDEN_DIM + config.HIDDEN_DIM // 4
        self.regressor = nn.Sequential(
            nn.Linear(combined_dim, config.HIDDEN_DIM // 2),
            nn.LayerNorm(config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM // 2, 1)      # выход одно значение
        )

    def forward(self, input_ids, attention_mask, image, mass, n_ingredients):
        
        text_features = self.text_model(input_ids, attention_mask).last_hidden_state[:, 0, :]
        image_features = self.image_model(image)

        text_emb = self.text_proj(text_features)
        image_emb = self.image_proj(image_features)

        fused_emb = text_emb * image_emb  # поэлементное умножение

        # обрабатываем числовые признаки
        numeric = torch.stack([mass, n_ingredients], dim=1)
        numeric_emb = self.numeric_proj(numeric)

        # конкатенируем
        combined = torch.cat([fused_emb, numeric_emb], dim=1)

        # регрессия
        pred = self.regressor(combined).squeeze(-1)

        return pred


def train(config, device):
    seed_everything(config.SEED)

    # Инициализация модели
    model = MultimodalModel(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

    set_requires_grad(model.text_model,
                      unfreeze_pattern=config.TEXT_MODEL_UNFREEZE, verbose=True)
    set_requires_grad(model.image_model,
                      unfreeze_pattern=config.IMAGE_MODEL_UNFREEZE, verbose=True)

    # Оптимизатор с разными LR
    optimizer = AdamW([{
        'params': model.text_model.parameters(),
        'lr': config.TEXT_LR
    }, {
        'params': model.image_model.parameters(),
        'lr': config.IMAGE_LR
    }, {
        'params': model.regressor.parameters(),
        'lr': config.CLASSIFIER_LR
    }, {
        'params': model.numeric_proj.parameters(),
        'lr': config.CLASSIFIER_LR
    }])

    # Объявляем loss = MSE
    criterion = nn.MSELoss()

    # Загрузка данных
    transforms = get_transforms(config)
    val_transforms = get_transforms(config, ds_type="val")
    train_dataset = MultimodalDataset(config, transforms)
    val_dataset = MultimodalDataset(config, val_transforms, ds_type="val")
    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE,
                              shuffle=True,
                              collate_fn=partial(collate_fn,
                                                 tokenizer=tokenizer))
    val_loader = DataLoader(val_dataset,
                            batch_size=config.BATCH_SIZE,
                            shuffle=False,
                            collate_fn=partial(collate_fn,
                                               tokenizer=tokenizer))

    # Инициализируем метрики регрессии
    mae_metric_train = torchmetrics.MeanAbsoluteError().to(device)
    rmse_metric_train = torchmetrics.MeanSquaredError(squared=False).to(device)

    best_val_mae = float('inf')

    # Для сохранения истории обучения
    history = {
        'train_loss': [], 'train_mae': [], 'train_rmse': [],
        'val_loss': [], 'val_mae': [], 'val_rmse': []
    }

    print("training started")
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0.0

        # обнуляем метрики перед эпохой
        mae_metric_train.reset()
        rmse_metric_train.reset()

        for batch in train_loader:
            # Подготовка данных
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['image'].to(device),
                'mass': batch['mass'].to(device),          
                'n_ingredients': batch['n_ingredients'].to(device) 
            }
            labels = batch['label'].to(device)  

            # Forward
            optimizer.zero_grad()
            preds = model(**inputs)            
            loss = criterion(preds, labels)

            # Backward
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Обновляем метрики
            mae_metric_train.update(preds, labels)
            rmse_metric_train.update(preds, labels)

        # Валидация
        val_loss, val_mae, val_rmse = validate(model, val_loader, device, criterion)

        # Вычисляем train метрики
        train_loss = total_loss / len(train_loader)
        train_mae = mae_metric_train.compute().cpu().numpy()
        train_rmse = rmse_metric_train.compute().cpu().numpy()

        # Сохраняем в историю
        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)
        history['train_rmse'].append(train_rmse)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_rmse'].append(val_rmse)

        print(
            f"Epoch {epoch}/{config.EPOCHS-1} | "
            f"Train Loss: {train_loss:.4f} | Train MAE: {train_mae:.4f} | Train RMSE: {train_rmse:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f}"
        )

        # Сохраняем лучшую модель по MAE
        if val_mae < best_val_mae:
            print(f"New best model, epoch: {epoch} (Val MAE: {val_mae:.4f})")
            best_val_mae = val_mae
            torch.save(model.state_dict(), config.SAVE_PATH)

    # Cохраним историю обучения
    np.savez('training_history.npz', **history)
    print("Training history saved to training_history.npz")


def validate(model, val_loader, device, criterion):

    model.eval()
    total_loss = 0.0
    mae_metric = torchmetrics.MeanAbsoluteError().to(device)
    rmse_metric = torchmetrics.MeanSquaredError(squared=False).to(device)

    with torch.no_grad():
        for batch in val_loader:
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['image'].to(device),
                'mass': batch['mass'].to(device),
                'n_ingredients': batch['n_ingredients'].to(device)
            }
            labels = batch['label'].to(device)

            preds = model(**inputs)
            loss = criterion(preds, labels)
            total_loss += loss.item()

            mae_metric.update(preds, labels)
            rmse_metric.update(preds, labels)

    avg_loss = total_loss / len(val_loader)
    mae = mae_metric.compute().cpu().numpy()
    rmse = rmse_metric.compute().cpu().numpy()

    return avg_loss, mae, rmse