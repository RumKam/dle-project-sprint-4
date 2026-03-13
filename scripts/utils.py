import os
import random
from functools import partial
import numpy as np
import timm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import torchmetrics
from transformers import AutoModel, AutoTokenizer
from scripts.dataset import MultimodalDataset, collate_fn, get_transforms
import pandas as pd


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
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
        text_hidden_size = self.text_model.config.hidden_size
        
        self.image_model = timm.create_model(
            config.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0
        )
        image_hidden_size = self.image_model.num_features

        self.text_proj = nn.Linear(text_hidden_size, config.HIDDEN_DIM)
        self.image_proj = nn.Linear(image_hidden_size, config.HIDDEN_DIM)

        self.numeric_proj = nn.Linear(2, config.HIDDEN_DIM // 4)
        self.numeric_norm = nn.LayerNorm(config.HIDDEN_DIM // 4)

        self.fusion_layer = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM * 2, config.HIDDEN_DIM),
            nn.GELU(),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.Dropout(config.DROPOUT)
        )

        combined_dim = config.HIDDEN_DIM + config.HIDDEN_DIM // 4
        self.regressor = nn.Sequential(
            nn.Linear(combined_dim, config.HIDDEN_DIM),
            nn.GELU(),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.GELU(),
            nn.LayerNorm(config.HIDDEN_DIM // 2),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM // 2, 1)
        )

    def forward(self, input_ids, attention_mask, image, mass, n_ingredients):

        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_output.last_hidden_state[:, 0, :]
        
        image_features = self.image_model(image)

        text_emb = self.text_proj(text_features)
        image_emb = self.image_proj(image_features)

        fused_emb = torch.cat([text_emb, image_emb], dim=1)
        fused_emb = self.fusion_layer(fused_emb)

        numeric = torch.stack([mass, n_ingredients], dim=1)
        numeric_emb = self.numeric_proj(numeric)
        numeric_emb = self.numeric_norm(numeric_emb)

        combined = torch.cat([fused_emb, numeric_emb], dim=1)

        # Регрессия
        pred = self.regressor(combined).squeeze(-1)

        return pred


def train(config, device):
    seed_everything(config.SEED)

    model = MultimodalModel(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

    # Разморозка слоев
    set_requires_grad(model.text_model, unfreeze_pattern=config.TEXT_MODEL_UNFREEZE, verbose=True)
    set_requires_grad(model.image_model, unfreeze_pattern=config.IMAGE_MODEL_UNFREEZE, verbose=True)

    optimizer = AdamW([
        {'params': model.text_model.parameters(), 'lr': config.TEXT_LR},
        {'params': model.image_model.parameters(), 'lr': config.IMAGE_LR},
        {'params': [
            *model.regressor.parameters(),
            *model.text_proj.parameters(),
            *model.image_proj.parameters(),
            *model.numeric_proj.parameters(),
            *model.numeric_norm.parameters(),
            *model.fusion_layer.parameters()
        ], 'lr': config.CLASSIFIER_LR}
    ], weight_decay=config.WEIGHT_DECAY)

    train_df = pd.read_csv(config.TRAIN_DF_PATH)
    steps_per_epoch = len(train_df) // config.BATCH_SIZE
    total_steps = steps_per_epoch * config.EPOCHS
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[config.TEXT_LR, config.IMAGE_LR, config.CLASSIFIER_LR],
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos',
        cycle_momentum=True
    )

    criterion = nn.MSELoss()

    transforms = get_transforms(config)
    val_transforms = get_transforms(config, ds_type="val")

    train_dataset = MultimodalDataset(config, transforms, ds_type="train")
    val_dataset = MultimodalDataset(config, val_transforms, ds_type="val")

    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE,
                              shuffle=True,
                              collate_fn=partial(collate_fn, tokenizer=tokenizer, config=config),
                              num_workers=config.NUM_WORKERS,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=config.BATCH_SIZE,
                            shuffle=False,
                            collate_fn=partial(collate_fn, tokenizer=tokenizer, config=config),
                            num_workers=config.NUM_WORKERS,
                            pin_memory=True)

    best_val_mae = float('inf')
    history = {'train_loss': [], 'train_mae': [], 'train_rmse': [],
               'val_loss': [], 'val_mae': [], 'val_rmse': [], 'lr': []}

    print("=" * 60)
    print(f"Начало обучения {config.IMAGE_MODEL_NAME} + {config.TEXT_MODEL_NAME}")
    print("=" * 60)
    
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0.0
        mae_metric_train = torchmetrics.MeanAbsoluteError().to(device)
        rmse_metric_train = torchmetrics.MeanSquaredError(squared=False).to(device)
        lr_values = []

        for batch in train_loader:
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['image'].to(device),
                'mass': batch['mass'].to(device),
                'n_ingredients': batch['n_ingredients'].to(device)
            }
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            preds = model(**inputs)
            loss = criterion(preds, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            lr_values.append(optimizer.param_groups[0]['lr'])

            total_loss += loss.item()
            
            # Денормализация для метрик
            preds_real = preds * batch['calorie_std'] + batch['calorie_mean']
            labels_real = labels * batch['calorie_std'] + batch['calorie_mean']
            
            mae_metric_train.update(preds_real, labels_real)
            rmse_metric_train.update(preds_real, labels_real)

        train_loss = total_loss / len(train_loader)
        train_mae = mae_metric_train.compute().cpu().numpy()
        train_rmse = rmse_metric_train.compute().cpu().numpy()
        avg_lr = np.mean(lr_values)

        val_loss, val_mae, val_rmse = validate(model, val_loader, device, criterion,
                                                train_dataset.calorie_mean,
                                                train_dataset.calorie_std)

        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)
        history['train_rmse'].append(train_rmse)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_rmse'].append(val_rmse)
        history['lr'].append(avg_lr)

        print(f"Epoch {epoch:2d}/{config.EPOCHS-1} | "
              f"Train Loss: {train_loss:8.2f} | Train MAE: {train_mae:6.2f} | "
              f"Val Loss: {val_loss:8.2f} | Val MAE: {val_mae:6.2f} | "
              f"LR: {avg_lr:.2e}")
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({
                'model_state_dict': model.state_dict(),
                'calorie_mean': train_dataset.calorie_mean,
                'calorie_std': train_dataset.calorie_std,
                'config': {
                    'IMAGE_MODEL_NAME': config.IMAGE_MODEL_NAME,
                    'TEXT_MODEL_NAME': config.TEXT_MODEL_NAME,
                    'HIDDEN_DIM': config.HIDDEN_DIM
                }
            }, config.SAVE_PATH)

    np.savez('training_history.npz', **history)


def validate(model, val_loader, device, criterion, calorie_mean, calorie_std):
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

            # Денормализация для метрик
            preds_real = preds * calorie_std + calorie_mean
            labels_real = labels * calorie_std + calorie_mean

            mae_metric.update(preds_real, labels_real)
            rmse_metric.update(preds_real, labels_real)

    avg_loss = total_loss / len(val_loader)
    mae = mae_metric.compute().cpu().numpy()
    rmse = rmse_metric.compute().cpu().numpy()

    return avg_loss, mae, rmse