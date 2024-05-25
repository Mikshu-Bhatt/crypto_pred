import requests
from bs4 import BeautifulSoup
from crypto_args import args
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau,CosineAnnealingLR,StepLR
import pandas as pd


rows = []
lab = []
coin_type = ['bitcoin','litecoin','bitcoin cash','dogecoin','dash','bitcoin gold','vertcoin']
for ct in coin_type:
    c = 0
    for i in range(120):
        try:
            url = f'https://bitinfocharts.com/top-100-richest-{ct}-addresses-{i}.html'  # Replace with your URL
            response = requests.get(url)

            soup = BeautifulSoup(response.content, 'html.parser')

            # Find the specific table by id or class
            table = soup.find('table', {'id':"tblOne2"})  # or {'id': 'specific-table-id'}
            # print(table)
            # Extract headers
            headers = []
            for th in table.find_all('th'):
                headers.append(th.text.strip())

            # Extract rows
            
            for tr in table.find_all('tr'):
                cells = tr.find_all(['td', 'th'])
                row = [cell.text.strip() for cell in cells]
                if row:  # Avoid empty rows
                    rows.append(row[1].split('Balance')[0].replace('..','').split(' ')[0])
                    c+=1
            
        except:
            pass
    # print(c,len(rows))
    lab.extend([ct]*c)
    

chars = sorted(list(set("".join(rows))))
char_to_index = {c: i for i, c in enumerate(chars)}

def address_to_sequence(address):
    return [char_to_index[c] for c in address]

sequences = [address_to_sequence(addr) for addr in rows]

max_length = max(len(seq) for seq in sequences)
padded_sequences = [seq + [0] * (max_length - len(seq)) for seq in sequences]

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(lab)

X_train, X_test, y_train, y_test = train_test_split(padded_sequences, encoded_labels, test_size=0.2, random_state=42)
lens = [len(i) for i in rows]
# Custom Dataset
class WalletDataset(Dataset):
    def __init__(self, sequences, labels,lens):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.lens = torch.tensor(lens, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], self.lens[idx]

train_dataset = WalletDataset(X_train, y_train,lens)
test_dataset = WalletDataset(X_test, y_test,lens)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)

class CharacterLevelModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, dropout_prob=0.2):
        super(CharacterLevelModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
#         self.dropout1 = nn.Dropout(dropout_prob)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout3 = nn.Dropout(dropout_prob)
        self.mlp = nn.Linear(output_size + 1, output_size)
        
    def forward(self, x, l):
        x = self.embedding(x)
#         x = self.dropout1(x)
        x, _ = self.lstm(x)
        x = self.dropout2(x)
        x = x[:, -1, :]  # Take the last hidden state
        x = self.fc(x)
        x = self.dropout3(x)
        x = torch.cat([x, l.unsqueeze(0).view(x.shape[0], 1)], dim=1)
        x = self.mlp(x)
        return x

# Hyperparameters
vocab_size = len(chars)
embed_size = 50
hidden_size = 128
output_size = len(set(lab))

# Initialize model, criterion, optimizer
model = CharacterLevelModel(vocab_size, embed_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay = 1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# Training Loop
num_epochs = 300
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
min_val_loss = np.inf
for epoch in range(num_epochs):
    model.train()
    tr_pred = []
    tr_label = []
    val_pred = []
    val_label = []
    uu = 0
    for sequences, labels, l in train_loader:
        uu +=1 
        sequences, labels , l= sequences.to(device), labels.to(device), l.to(device)
        
        optimizer.zero_grad()
        output = model(sequences,l)
        loss = criterion(output, labels)
        _, predicted = torch.max(output.data, 1)
        tr_pred.extend(predicted.cpu().numpy().astype(int))
        tr_label.extend(labels.cpu().numpy())
        loss.backward()
        optimizer.step()
        if uu == 300 and epoch % 30 == 0:
            print(predicted,labels)
#         print(loss.item())
    scheduler.step()
    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for sequences, labels, l in test_loader:
            sequences, labels , l= sequences.to(device), labels.to(device), l.to(device)
            output = model(sequences,l)
            loss_val = criterion(output, labels)
            _, predicted = torch.max(output.data, 1)
            val_pred.extend(predicted.cpu().numpy().astype(int))
            val_label.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    train_report=classification_report(np.array(tr_label),np.array(tr_pred),output_dict=True, zero_division=0)
    val_report=classification_report(np.array(val_label),np.array(val_pred),output_dict=True, zero_division=0)
    if epoch % 12 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, tr Loss: {loss.item()}, val loss {loss_val},Accuracy: {accuracy}')
        df_tr = pd.DataFrame(train_report)
        df_val = pd.DataFrame(val_report)
        print('for training')
        print()
        print(f'  {df_tr}')
        print()
        print()
        print(' for val')
        print()
        print(f' {df_val}')
    if min_val_loss > loss_val:
        min_val_loss = loss_val
        torch.save(model.state_dict(), args.model_output)