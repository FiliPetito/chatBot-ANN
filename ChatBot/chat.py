import random
import json

import torch

from datetime import datetime

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize_and_correct
from spellchecker import SpellChecker

device = torch.device('cpu')
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r', encoding="utf8") as json_data:
    intents = json.load(json_data)

# Inizializza il correttore ortografico
spell = SpellChecker(language='it')

# Funzione per correggere l'ortografia dell'input
def correct_spelling(sentence):
    corrected_sentence = []
    for word in sentence:
        corrected_word = spell.correction(word)  # Corregge la parola
        corrected_sentence.append(corrected_word if corrected_word else word)
    return corrected_sentence

FILE = "data.pth"
#data = torch.load(FILE)
data = torch.load(FILE, map_location=torch.device('cpu'))

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Cardinal"

def get_response(msg):
    # Tokenizza e corregge l'ortografia dell'input utente
    sentence = tokenize_and_correct(msg)
    sentence = correct_spelling(sentence)  # Applica il correttore ortografico
    print("Correct sentence ",  sentence )
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        if tag == "orario":
            return "Oggi è: " + str(datetime.now())
        else:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    return random.choice(intent['responses'])
    
    return "Non ho capito, potresti riprovare ?"

    