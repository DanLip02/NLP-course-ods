"""
Улучшенное решение для классификации отзывов (1-5 звёзд)
Метрика: macro F1

Что исправлено по сравнению с baseline:
1. КРИТИЧЕСКИЙ БАГ: test_size=0.85 → обучалось только на 15% данных!
2. LR 3e-4 → 2e-5 (правильный LR для BERT fine-tuning)
3. rubert-tiny2 (312 dim) → rubert-base-cased (768 dim) — в 2.5x мощнее
4. Добавлен AdamW + линейный warmup scheduler
5. Добавлены веса классов для борьбы с дисбалансом (класс 5 в 6x больше класса 2)
6. 3 эпохи вместо 2
7. Метрика F1 вместо Accuracy
8. MAX_LEN 128 → 256 (отзывы могут быть длиннее)
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from tqdm import tqdm

# ---------------------------
# Константы
# ---------------------------
MODEL_NAME = "DeepPavlov/rubert-base-cased"
MAX_LEN = 256
BATCH_SIZE = 32
N_EPOCHS = 3
LR = 2e-5
WARMUP_RATIO = 0.1
DROPOUT = 0.1
N_CLASSES = 5


# ---------------------------
# Dataset (исправленный __getitem__)
# ---------------------------
class ReviewDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len, labels=None):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        text = " ".join(text.split())  # нормализация пробелов

        # Современный способ токенизации (вместо encode_plus)
        enc = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        item = {
            "ids": enc["input_ids"].squeeze(0),   # убираем batch dimension
            "mask": enc["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["targets"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ---------------------------
# Модель
# ---------------------------
class BertClassifier(torch.nn.Module):
    def __init__(self, model_name, n_classes, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(hidden_size, n_classes)

    def forward(self, ids, mask):
        out = self.bert(input_ids=ids, attention_mask=mask)
        cls = out.last_hidden_state[:, 0, :]  # CLS токен
        cls = self.dropout(cls)
        return self.classifier(cls)


# ---------------------------
# Функции обучения и валидации
# ---------------------------
def train_epoch(model, loader, optimizer, scheduler, loss_fn, device):
    model.train()
    losses = []
    for batch in tqdm(loader, desc="Train"):
        ids = batch["ids"].to(device)
        mask = batch["mask"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()
        logits = model(ids, mask)
        loss = loss_fn(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
    return np.mean(losses)


def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    all_preds, all_labels, losses = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val"):
            ids = batch["ids"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["targets"].to(device)
            logits = model(ids, mask)
            loss = loss_fn(logits, targets)
            losses.append(loss.item())
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(targets.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average="macro")
    return np.mean(losses), f1


# ---------------------------
# Основной блок
# ---------------------------
if __name__ == "__main__":
    # Устройство
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
    print(f"Using device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")

    # Путь к данным (укажите свой)
    DATA_PATH = r"/Users/danilalipatov/nlp_huawei_new2_task"

    # Загрузка данных
    train_df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))

    # Кодирование меток
    le = LabelEncoder()
    train_df["label"] = le.fit_transform(train_df["rate"])  # 1-5 → 0-4

    # Веса классов (обработка дисбаланса)
    class_counts = np.bincount(train_df["label"])
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * N_CLASSES
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    print("Class weights:", dict(zip(le.classes_, class_weights.round(3))))

    # Разделение на train/val
    train_data, val_data = train_test_split(
        train_df, test_size=0.15, random_state=42, stratify=train_df["label"]
    )
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Токенизатор
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Датасеты
    train_dataset = ReviewDataset(
        train_data["text"].values, tokenizer, MAX_LEN, train_data["label"].values
    )
    val_dataset = ReviewDataset(
        val_data["text"].values, tokenizer, MAX_LEN, val_data["label"].values
    )
    test_dataset = ReviewDataset(test_df["text"].values, tokenizer, MAX_LEN)

    # DataLoader'ы
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Модель
    model = BertClassifier(MODEL_NAME, N_CLASSES, DROPOUT).to(DEVICE)

    # Оптимизатор и планировщик
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    total_steps = len(train_loader) * N_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    loss_fn = CrossEntropyLoss(weight=class_weights_tensor)

    # Обучение
    best_f1 = 0
    best_model_path = "best_model.pt"

    for epoch in range(1, N_EPOCHS + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{N_EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, DEVICE)
        val_loss, val_f1 = eval_epoch(model, val_loader, loss_fn, DEVICE)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ Best model saved (F1={best_f1:.4f})")

    print(f"\nBest Val F1: {best_f1:.4f}")

    # Предсказание на тестовой выборке
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    predictions = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predict"):
            ids = batch["ids"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            logits = model(ids, mask)
            predictions.extend(logits.argmax(1).cpu().numpy())

    predictions = np.array(predictions)
    decoded = le.inverse_transform(predictions)  # обратно в 1-5

    # Сохранение сабмишена
    submission = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"))
    submission["rate"] = decoded
    submission.to_csv("submission.csv", index=False)
    print("\nsubmission.csv saved!")
    print(submission["rate"].value_counts().sort_index())