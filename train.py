from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from backend.models.sentiment_net import SentimentLSTM
from backend.services.text import tokenize


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "datasets" / "IMDB_Dataset.csv"
ARTIFACT_DIR = ROOT / "models"


class ReviewDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], vocab: dict[str, int], max_len: int) -> None:
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids = [self.vocab.get(token, 1) for token in tokenize(self.texts[index])][: self.max_len]
        ids += [0] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[index], dtype=torch.float32)


def build_vocab(texts: list[str], vocab_size: int) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize(text))
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, _ in counter.most_common(vocab_size - 2):
        vocab[word] = len(vocab)
    return vocab


def train_lstm(train_loader: DataLoader, valid_loader: DataLoader, vocab_size: int, epochs: int, device: torch.device) -> None:
    model = SentimentLSTM(vocab_size=vocab_size).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    best_acc = 0.0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                probs = torch.sigmoid(model(inputs))
                correct += ((probs >= 0.5).float() == labels).sum().item()
                total += labels.numel()
        acc = correct / max(total, 1)
        print(f"LSTM epoch {epoch}/{epochs} loss={total_loss / max(len(train_loader), 1):.4f} val_acc={acc:.4f}")
        if acc >= best_acc:
            best_acc = acc
            best_state = model.state_dict()

    if best_state is not None:
        ARTIFACT_DIR.mkdir(exist_ok=True)
        torch.save(best_state, ARTIFACT_DIR / "sentiment_lstm.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PyTorch LSTM sentiment model on IMDB reviews.")
    parser.add_argument("--sample-size", type=int, default=12000, help="Rows to train on. Use 50000 for the full dataset.")
    parser.add_argument("--vocab-size", type=int, default=20000)
    parser.add_argument("--max-len", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    df = pd.read_csv(DATASET_PATH)
    if args.sample_size and args.sample_size < len(df):
        df = df.sample(args.sample_size, random_state=42)
    texts = df["review"].astype(str).tolist()
    labels = (df["sentiment"].str.lower() == "positive").astype(int).tolist()

    train_texts, valid_texts, train_labels, valid_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    vocab = build_vocab(train_texts, args.vocab_size)
    train_ds = ReviewDataset(train_texts, train_labels, vocab, args.max_len)
    valid_ds = ReviewDataset(valid_texts, valid_labels, vocab, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size)

    ARTIFACT_DIR.mkdir(exist_ok=True)
    (ARTIFACT_DIR / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (ARTIFACT_DIR / "sentiment_meta.json").write_text(
        json.dumps({"max_len": args.max_len, "vocab_size": len(vocab)}),
        encoding="utf-8",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} with {len(train_ds)} train and {len(valid_ds)} validation reviews.")
    train_lstm(train_loader, valid_loader, len(vocab), args.epochs, device)


if __name__ == "__main__":
    main()
