import torch
from torch.utils.data import Dataset
from PIL import Image
import timm
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import albumentations as A
import random


class MultimodalDataset(Dataset):

    def __init__(self, config, transforms, ds_type="train"):
        if ds_type == "train":
            self.df = pd.read_csv(config.TRAIN_DF_PATH)
        else:
            self.df = pd.read_csv(config.VAL_DF_PATH)
        
        self.image_cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
        self.transforms = transforms
        self.ds_type = ds_type
        self.use_text_aug = config.USE_TEXT_AUG and ds_type == "train"
        
        self.calorie_mean = self.df['total_calories'].mean()
        self.calorie_std = self.df['total_calories'].std()
        self.df['total_calories_norm'] = (self.df['total_calories'] - self.calorie_mean) / self.calorie_std
        
        self.mass_mean = self.df['total_mass'].mean()
        self.mass_std = self.df['total_mass'].std()
        self.n_ingr_mean = self.df['n_ingredients'].mean()
        self.n_ingr_std = self.df['n_ingredients'].std()

    def __len__(self):
        return len(self.df)

    def _augment_text(self, text):
        """Простая аугментация: случайное удаление слов (dropout)"""
        if not self.use_text_aug or text == "deprecated" or not isinstance(text, str):
            return text
        
        words = text.split(',')
        # С вероятностью 10% удаляем случайное слово
        if len(words) > 3 and random.random() < 0.1:
            idx_to_remove = random.randint(0, len(words) - 1)
            words.pop(idx_to_remove)
        
        return ','.join(words)

    def __getitem__(self, idx):
        text = self.df.loc[idx, "ingredients"]
        
        # Применяем аугментацию текста только для train
        text = self._augment_text(text)
        
        label = self.df.loc[idx, "total_calories_norm"]
        mass = (self.df.loc[idx, "total_mass"] - self.mass_mean) / self.mass_std
        n_ingredients = (self.df.loc[idx, "n_ingredients"] - self.n_ingr_mean) / self.n_ingr_std
        dish_id = self.df.loc[idx, "dish_id"]

        img_id = dish_id
        
        try:
            image = Image.open(f"data/images/{img_id}/rgb.png").convert('RGB')
        except FileNotFoundError:
            image = Image.new('RGB', (224, 224), color='black')

        image = self.transforms(image=np.array(image))["image"]

        return {
            "label": torch.tensor(label, dtype=torch.float32), 
            "image": image,
            "text": text,
            "mass": torch.tensor(mass, dtype=torch.float32),
            "n_ingredients": torch.tensor(n_ingredients, dtype=torch.float32),
            "dish_id": dish_id,
            "calorie_mean": self.calorie_mean,
            "calorie_std": self.calorie_std
        }


def collate_fn(batch, tokenizer, config):
    texts = [item["text"] for item in batch]
    images = torch.stack([item["image"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    dish_ids = [item["dish_id"] for item in batch]

    masses = torch.stack([item["mass"] for item in batch])
    n_ingredients = torch.stack([item["n_ingredients"] for item in batch])
    
    calorie_mean = batch[0]["calorie_mean"]
    calorie_std = batch[0]["calorie_std"]

    tokenized_input = tokenizer(texts,
                                return_tensors="pt",
                                padding="max_length",
                                truncation=True,
                                max_length=config.MAX_LENGTH)

    return {
        "label": labels,               
        "image": images,
        "mass": masses,                 
        "n_ingredients": n_ingredients,    
        "dish_id": dish_ids,
        "input_ids": tokenized_input["input_ids"],
        "attention_mask": tokenized_input["attention_mask"],
        "calorie_mean": calorie_mean,
        "calorie_std": calorie_std
    }


def get_transforms(config, ds_type="train"):
    cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)

    if ds_type == "train":
        transforms = A.Compose(
            [
                A.RandomResizedCrop(size=(cfg.input_size[1], cfg.input_size[2]),
                                    scale=(0.8, 1.0), p=1.0),
                A.Rotate(limit=30, p=0.7),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.ToTensorV2(p=1.0)
            ],
            seed=42,
        )
    else:
        transforms = A.Compose(
            [
                A.SmallestMaxSize(
                    max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0),
                A.CenterCrop(
                    height=cfg.input_size[1], width=cfg.input_size[2], p=1.0),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.ToTensorV2(p=1.0)
            ],
            seed=42,
        )

    return transforms