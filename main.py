import re
from collections import Counter
from datasets import load_dataset
from torch.utils.data import Dataset
from torch.utils.data import random_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# We need to map words to indices in order to create embedding ...we neeed
# to encode words into numbers to be put in vector space


def create_vocabulary(ds, max_words=10000):
    word2idx = {
        "<PAD>": 0,
        "<UNK>": 1,
    }
    words = []
    for example in ds:
        text = example["sms"]
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        words.extend(text.split())

    word_counts = Counter(words)
    common_words = word_counts.most_common(max_words - 2)
    for word, _ in common_words:
        word2idx[word] = len(word2idx)

    return word2idx


def create_splits(ds):
    # 80/20 split
    full_dataset = ds['train']
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    return train_dataset, test_dataset


class SMSDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_length=100):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx].lower()
        text = re.sub(r"[^\w\s]", "", text)
        words = text.split()

        indices = [self.word2idx.get(
            word, self.word2idx["<UNK>"]) for word in words]

        if len(indices) < self.max_length:
            indices.extend([self.word2idx["<PAD>"]] *
                           (self.max_length - len(indices)))
        else:
            indices = indices[:self.max_length]

        return {
            'text': torch.tensor(indices, dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=200, hidden_dim=256):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        # Dropout layer
        self.dropout = nn.Dropout(0.4)

        # Additional dense layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        embedded = self.embedding(x)

        lstm_out, (hidden, cell) = self.lstm(embedded)

        # Concatenate forward and backward hidden states
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        hidden = self.dropout(hidden)

        # Additional layer with ReLU activation
        hidden = torch.relu(self.fc1(hidden))
        hidden = self.dropout(hidden)

        # Final classification layer
        out = self.fc2(hidden)
        return out


def create_loaders(ds, train_dataset, test_dataset, vocab):
    train_texts = [ds['train'][idx]['sms']
                   for idx in train_dataset.indices]
    train_labels = [ds['train'][idx]['label'] for idx in train_dataset.indices]

    test_texts = [ds['train'][idx]['sms'] for idx in test_dataset.indices]
    test_labels = [ds['train'][idx]['label'] for idx in test_dataset.indices]

    train_dataset = SMSDataset(train_texts, train_labels, vocab)
    test_dataset = SMSDataset(test_texts, test_labels, vocab)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128)
    return train_loader, test_loader


def create_model(vocab_size, device):
    model = LSTMClassifier(vocab_size, 100)
    model = model.to(device)
    return model


def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_correct = 0
    total = 0

    # Track spam and ham separately
    spam_correct = 0
    spam_total = 0
    ham_correct = 0
    ham_total = 0

    with torch.no_grad():
        for batch in test_loader:
            texts = batch['text'].to(device)
            labels = batch['label'].to(device)

            outputs = model(texts)
            _, predicted = torch.max(outputs, 1)

            # Overall metrics
            total += labels.size(0)
            total_correct += (predicted == labels).sum().item()

            # Spam metrics
            spam_mask = labels == 1
            spam_total += spam_mask.sum().item()
            spam_correct += ((predicted == labels) & spam_mask).sum().item()

            # Ham metrics
            ham_mask = labels == 0
            ham_total += ham_mask.sum().item()
            ham_correct += ((predicted == labels) & ham_mask).sum().item()

    return {
        'overall_acc': total_correct / total,
        'spam_acc': spam_correct / spam_total if spam_total > 0 else 0,
        'ham_acc': ham_correct / ham_total if ham_total > 0 else 0
    }


def train(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=5):
    best_accuracy = 0
    for epoch in range(num_epochs):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device)

        test_results = evaluate(
            model, test_loader, criterion, device)

        test_accuracy = test_results['overall_acc']

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
        print(f"test_results: {test_results}")

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), 'best_model.pth')

        print(f"Best accuracy: {best_accuracy:.4f}")


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        # batch data
        texts = batch['text'].to(device)
        labels = batch['label'].to(device)

        # forward pass
        outputs = model(texts)
        loss = criterion(outputs, labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def main():
    ds = load_dataset("ucirvine/sms_spam")
    train_dataset, test_dataset = create_splits(ds)
    vocab = create_vocabulary(train_dataset)

    train_loader, test_loader = create_loaders(
        ds, train_dataset, test_dataset, vocab)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = create_model(len(vocab), device)

    # Calculate weights based on inverse class frequency
    n_spam = 747
    n_ham = 4827
    total = n_spam + n_ham

    weight_spam = total / (2 * n_spam)  # More weight to spam class
    weight_ham = total / (2 * n_ham)    # Less weight to ham class

    class_weights = torch.FloatTensor([weight_ham, weight_spam]).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001)

    train(model, train_loader, test_loader, criterion, optimizer, device)


if __name__ == "__main__":
    main()
