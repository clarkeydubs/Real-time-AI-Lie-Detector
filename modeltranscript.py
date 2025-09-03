import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import random
from collections import Counter
import pickle

#To stop training when called by Late Fusion
if __name__ == "__main__":
    
    #Load transcript data from txt file
    data_path = "transcriptdata.txt"
    data = []

    #Split each record by # (commas could be used within the transcript) and append to dataset list.
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or '#' not in line:
                continue
            text, label = line.rsplit("#", 1)
            label = label.strip().lower()
            if label in ["truth", "lie"]:
                #Normalise label 
                normalised_label = "truthful" if label in ["truth"] else "deceptive"
                data.append((text.strip(), normalised_label))

    #print(f"Loaded {len(data)} labeled samples.") DEBUG

    #Shuffle and split list
    random.shuffle(data)
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    label_map = {"truthful": 0, "deceptive": 1}

    #Tokeniser
    def tokeniser(text):
        return text.lower().split()

    #Vocabulary builder
    def build_vocab(data):
        counter = Counter()
        for text, _ in data:
            counter.update(tokeniser(text))
        vocab = {"<pad>": 0, "<unk>": 1}
        for i, word in enumerate(counter.keys(), start=2):
            vocab[word] = i
        return vocab

    #Encoder
    def encode(tokens):
        return [vocab.get(token, vocab["<unk>"]) for token in tokens]

    #Use vocab builder
    vocab = build_vocab(data)

    #########################################################
    #####           Transcript Text Dataset             #####
    #########################################################

    class TextDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            text, label = self.data[idx]
            tokens = tokeniser(text)
            token_ids = torch.tensor(encode(tokens), dtype=torch.long)
            label_tensor = torch.tensor(label_map[label], dtype=torch.long)
            return token_ids, label_tensor

    def collate_batch(batch):
        text_batch, label_batch = zip(*batch)
        text_batch = pad_sequence(text_batch, batch_first=True, padding_value=vocab["<pad>"])
        label_batch = torch.tensor(label_batch, dtype=torch.long)
        return text_batch, label_batch

    train_loader = DataLoader(TextDataset(data), batch_size=2, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(TextDataset(test_data), batch_size=4, shuffle=False, collate_fn=collate_batch)

#######################################################################
#######                     Model Definition                    #######
#######################################################################

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, return_features=False):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        features = hidden[-1]
        if return_features:
            return features
        return self.fc(features)


if __name__ == "__main__":

    ########################################################
    #########             Training                ##########
    ########################################################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TextClassifier(len(vocab), embed_dim=64, hidden_dim=128, output_dim=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    train_accuracies = []
    test_accuracies = []
    test_losses = []

    for epoch in range(15):
        model.train()
        correct = 0
        total = 0
        epoch_loss = 0

        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = epoch_loss
        train_acc = correct / total * 100
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

    #Save model
    torch.save(model.state_dict(), "text_model.pth") 
    
    #Save vocab after training 
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    ########################################
    ######         Evaluation         ######
    ########################################

    model.eval()
    correct = 0
    total = 0
    epoch_loss = 0

    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)

            epoch_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_loss = epoch_loss
    test_acc = correct / total * 100
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    #Log
    print(f"Epoch {epoch+1}: "
        f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
        f"Val Loss={test_loss:.4f}, Val Acc={test_acc:.2f}%")
