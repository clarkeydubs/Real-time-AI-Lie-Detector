import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import os
import random
from collections import Counter
import pickle

#To stop training when called by Late Fusion
if __name__ == "__main__":
    
    #Tokeniser
    def tokeniser(text):
        return text.lower().split()

    #Get vocab from pkl file
    def build_vocab(data):
        with open("vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        return vocab

    def encode(tokens, vocab):
        return [vocab.get(token, vocab["<unk>"]) for token in tokens]

    ############################################
    ####          Fusion Dataset            ####
    ############################################

    class FusionDataset(Dataset):
        def __init__(self, data, vocab, gesture_df, audio_feature_dir, label_map):
            self.data = data
            self.vocab = vocab
            self.gesture_df = gesture_df
            self.audio_feature_dir = audio_feature_dir
            self.label_map = label_map

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            text, label, sample_id = self.data[idx]
            tokens = tokeniser(text)
            token_ids = torch.tensor(encode(tokens, self.vocab), dtype=torch.long)
            gesture_features = torch.tensor(self.gesture_df.loc[sample_id].values.astype('float32'))
            audio_path = os.path.join(self.audio_feature_dir, f"{sample_id}.pt")
            audio_features = torch.load(audio_path)
            return token_ids, gesture_features, audio_features, torch.tensor(self.label_map[label])

    #Collate Function
    def collate_fn(batch):
        texts, gestures, audios, labels = zip(*batch)
        text_batch = pad_sequence(texts, batch_first=True, padding_value=0)
        gesture_batch = torch.stack(gestures)
        audio_batch = torch.stack(audios)
        label_batch = torch.tensor(labels)
        return text_batch, gesture_batch, audio_batch, label_batch

###################################################################################
#######                         Model Definition                            #######
###################################################################################

class LateFusionModel(nn.Module):
    def __init__(self, text_model, audio_model, gesture_model, fusion_hidden_dim=128):
        super().__init__()
        self.text_model = text_model
        self.audio_model = audio_model
        self.gesture_model = gesture_model
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128 + 128, fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim, 2)
        )

    def forward(self, text_input, gesture_input, audio_input):
        text_feat = self.text_model(text_input, return_features=True)
        gesture_feat = self.gesture_model(gesture_input)
        audio_feat = self.audio_model(audio_input, return_features=True)

        if audio_feat.dim() == 3:
            audio_feat = audio_feat.mean(dim=1)  #Average over time

        return self.fusion(torch.cat((text_feat, gesture_feat, audio_feat), dim=1))

#To stop training when called by Late Fusion
if __name__ == "__main__":
    ###############################################################
    #######              Dataset & DataLoading              #######
    ###############################################################

    #initialise data list
    data = []

    #Transcription Dataset
    with open("transcriptdata.txt", "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("#")
            if len(parts) != 3:
                continue
            filename, text, label = parts
            sample_id = os.path.splitext(filename.strip())[0]  # Removes '.txt'
            label = label.strip().lower()
            normalized_label = "truthful" if label in ["truth", "truthful"] else "deceptive"
            data.append((text.strip(), normalized_label, sample_id))

    random.shuffle(data)
    split_idx = int(0.8 * len(data))
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    vocab = build_vocab(train_data)
    label_map = {"truthful": 0, "deceptive": 1}

    #Gesture Dataset
    gesture_df = pd.read_csv("Real-life_Deception_Detection_2016/Annotation/All_Gestures.csv")
    gesture_df = gesture_df.drop(columns=["class"])
    gesture_df["id"] = gesture_df["id"].apply(lambda x: os.path.splitext(x)[0])

    gesture_df = gesture_df.set_index("id")

    #Audio Feature Directory
    audio_feature_dir = "audio_features"

    train_dataset = FusionDataset(train_data, vocab, gesture_df, audio_feature_dir, label_map)
    test_dataset = FusionDataset(test_data, vocab, gesture_df, audio_feature_dir, label_map)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    ########################################################################################
    ##########           Define Sub-Models and Load Pretrained Weights           ###########
    ########################################################################################

    from modeltranscript import TextClassifier  
    from modelgestures import MultiHotClassifier
    from modelaudio import AudioClassifier

    text_model = TextClassifier(vocab_size=len(vocab), embed_dim=64, hidden_dim=128, output_dim=2, padding_idx=vocab["<pad>"])
    gesture_model = MultiHotClassifier(input_dim=gesture_df.shape[1])
    audio_model = AudioClassifier(input_dim=40, output_dim=128)

    #Load pretrained weights
    if os.path.exists("text_model.pth"):
        text_model.load_state_dict(torch.load("text_model.pth"))
        print("Loaded text model weights")

    if os.path.exists("gesture_model.pth"):
        gesture_model.load_state_dict(torch.load("gesture_model.pth"))
        print("Loaded gesture model weights")

    if os.path.exists("audio_model.pth"):
        audio_model.load_state_dict(torch.load("audio_model.pth"))
        print("Loaded audio model weights")

    #######################################################################
    #########                     Training                        #########
    #######################################################################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LateFusionModel(text_model, audio_model, gesture_model).to(device)
    criterion = nn.CrossEntropyLoss()

    #Only update trainable parameters
    optimiser = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    for epoch in range(15):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for text, gesture, audio, labels in train_loader:
            text, gesture, audio, labels = text.to(device), gesture.to(device), audio.to(device), labels.to(device)
            outputs = model(text, gesture, audio)
            loss = criterion(outputs, labels)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / total * 100
        print(f"Epoch {epoch+1}: Train Loss={running_loss:.4f}, Accuracy={acc:.2f}%")

    #Save Fusion Model
    torch.save(model.state_dict(), "fusion_model.pth")
    print("Saved fusion model")
