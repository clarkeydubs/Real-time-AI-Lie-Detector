import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

#To stop training when called by Late Fusion
if __name__ == "__main__":
    
    #Retrieve and initialise df w/ relative hot encoded gesture dataset
    df = pd.read_csv("Real-life_Deception_Detection_2016\Annotation\All_Gestures.csv")  

    #Drop id
    df = df.drop(columns=["id"])

    #Encode labels
    label_encoder = LabelEncoder()
    df["class"] = label_encoder.fit_transform(df["class"])  

    #Split into features and labels
    X = df.drop(columns=["class"]).values.astype('float32')
    y = df["class"].values.astype('int64')

    #Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    #######################
    ### PyTorch Dataset ###
    #######################

    class MultiHotDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X)
            self.y = torch.tensor(y)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    train_dataset = MultiHotDataset(X_train, y_train)
    test_dataset = MultiHotDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

################################
####### Model Definition #######
################################

class MultiHotClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim) 
        )

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    
    #Instantiate model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Additional head for classification
    model = MultiHotClassifier(X.shape[1]).to(device)
    
    #Train
    classifier_head = nn.Linear(128, 1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

    #Initialise Metric vars
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []


    for epoch in range(12):
        model.train()
        classifier_head.train()
        total_loss = 0
        correct = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            features = model(inputs) 
            logits = classifier_head(features)  
            
            loss = criterion(logits.squeeze(1), labels.float())

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            total_loss += loss.item()

            predicted = (torch.sigmoid(logits) > 0.5).int().squeeze()
            correct += (predicted == labels).sum().item()

        train_acc = correct / len(train_dataset) * 100
        train_losses.append(total_loss)
        train_accuracies.append(train_acc)

        print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Accuracy={train_acc:.2f}%")


    #Save model for late fusion prediction
    torch.save(model.state_dict(), "gesture_model.pth")    

    #Evaluation
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(1), labels.float())
            test_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()

    test_acc = correct / len(test_dataset) * 100
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
