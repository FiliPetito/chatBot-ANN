import nltk
import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, lemmatize, tokenize_and_correct
from model import NeuralNet

from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords

nltk.download('stopwords')

# Definisci le stopwords
STOP_WORDS = set(stopwords.words('italian'))

with open('intents.json', 'r', encoding="utf8") as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize_and_correct(pattern)  # Tokenizza la frase
        all_words.extend(w)    # Aggiungi le parole alla lista
        xy.append((w, tag))    # Aggiungi il pattern e l'etichetta

all_words = [lemmatize(w)[0] for w in all_words if w not in STOP_WORDS and lemmatize(w)]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique lemmatized words:", all_words)



# Creazione dei dati di addestramento
X_train = []
y_train = []



for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Divide i dati in training e validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Parametri del modello
num_epochs = 500
batch_size = 8
learning_rate = 0.0005
input_size = len(X_train[0])
hidden_size = 32
output_size = len(tags)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Definisci il modello NeuralNet
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss e ottimizzatore
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        # Backward e ottimizzazione
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    #if (epoch+1) % 100 == 0:
    #    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    # Calcolo della perdita sul validation set
    with torch.no_grad():
        model.eval()
        val_outputs = model(torch.tensor(X_val, dtype=torch.float32).to(device))
        val_loss = criterion(val_outputs, torch.tensor(y_val, dtype=torch.long).to(device))
        model.train()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

# Salvataggio del modello e dei dati
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')