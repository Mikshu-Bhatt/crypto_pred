import sys
import json
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from crypto_args import args
import os


chars = [' ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'] # This should be saved/loaded in a real deployment
# print(chars)
char_to_index = {c: i for i, c in enumerate(chars)}
def address_to_sequence(address):
    return [char_to_index[c] for c in address]

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
        x = x[:, -1, :]  
        x = self.fc(x)
        x = self.dropout3(x)
        x = torch.cat([x, l.unsqueeze(0).view(x.shape[0], 1)], dim=1)
        x = self.mlp(x)
        return x

# Load the trained model
def load_model(vocab_size, embed_size, hidden_size, output_size, model_path):
    model = CharacterLevelModel(vocab_size, embed_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Function to process input address
def process_address(address, char_to_index):

    try:
        sequence = [address_to_sequence(char) for char in address]  

        max_length = max(len(seq) for seq in sequence)
        lens = [len(i) for i in sequence]
        padded_sequences = [seq + [0] * (max_length - len(seq)) for seq in sequence]
        
        return torch.tensor(padded_sequences),torch.tensor(lens, dtype=torch.long)   
    except Exception as e:
        print(f"Error processing address: {e}", file=sys.stderr)
        return None


def main():
    

    vocab_size = len(chars)
    embed_size = 50
    hidden_size = 128
    output_size = 7
    

    model_path = args.model_path
    
    reverse_dict = {0:'bitcoin',1:'litecoin',2:'bitcoin cash',3:'dogecoin',4:'dash',5:'bitcoin gold',6:'vertcoin'}
    model = load_model(vocab_size, embed_size, hidden_size, output_size, model_path)
    
    # Read input from STDIN
    try:
        for line in sys.stdin:
            address = line.strip().replace('[','').replace(']','').split(',')
            
            if not address:
                continue
            
            # Process the address
            input_tensor,l = process_address(address, char_to_index)
            if input_tensor is None:
                continue
            
            # Predict
            with torch.no_grad():
                output = model(input_tensor,l)
                _, predicted = torch.max(output, 1)
            predicted_label = predicted.cpu().numpy()
            final_op = [reverse_dict[predicted_label[i]] for i in range(len(predicted_label))]
            print(final_op)
            item = {'address' : address, 'crypto':final_op }
            df = pd.DataFrame(item)
            df.to_csv(os.path.join(args.output_dir,'final_op.csv'))
    
    except Exception as e:
        print(f"Error in main loop: {e}", file=sys.stderr)
        pass
if __name__ == "__main__":
    main()
