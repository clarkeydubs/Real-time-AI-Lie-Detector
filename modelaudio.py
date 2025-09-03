import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

#To stop training when called by Late Fusion
if __name__ == "__main__":
    
    #Config label & model device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #################################
    ##### Audio Feature Dataset #####
    #################################

    class AudioPTDataset(Dataset):
        def __init__(self, pt_files):
            self.pt_files = pt_files

        def __len__(self):
            return len(self.pt_files)

        def __getitem__(self, idx):
            filepath = self.pt_files[idx]
            features = torch.load(filepath)

            # Ensure features are 1D (flatten if needed)
            if features.ndim > 1:
                features = features.view(-1)

            filename = os.path.basename(filepath).lower()
            label = 1 if "lie" in filename or "deceptive" in filename else 0
            return features.float(), torch.tensor(label, dtype=torch.long)


    #Load all .pt files from relative directory
    all_files = [os.path.join("datasets/audio_features", f) for f in os.listdir("audio_features") if f.endswith(".pt")]
    train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)

    train_dataset = AudioPTDataset(train_files)
    test_dataset = AudioPTDataset(test_files)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

##################################################################################
#######                        Model Definition                            #######
##################################################################################

class AudioClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64,output_dim=128):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
        self.classifier = nn.Linear(output_dim, 2)  # 

    def forward(self, x, return_features=False):
        features = self.feature_extractor(x)
        if return_features:
            return features
        return self.classifier(features)

if __name__ == "__main__":
    
    #Get input dimension from one sample
    sample_feature, _ = train_dataset[0]
    input_dim = sample_feature.shape[0]

    model = AudioClassifier(input_dim=input_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

    ##################################################
    #########            Training             ########
    ##################################################

    for epoch in range(15):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / total * 100
        print(f"Epoch {epoch+1}: Train Loss={total_loss:.4f}, Train Acc={acc:.2f}%")

    #Save model for late fusion
    torch.save(model.state_dict(), "audio_model.pth") 

    ###############################################
    #####             Evaluation              #####
    ###############################################

    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"\nTest Accuracy: {correct / total * 100:.2f}%")
