import random
import json

import torch

from datetime import datetime

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize_and_correct
from spellchecker import SpellChecker

import spacy
import re

# Carica il modello di SpaCy per l'italiano
nlp = spacy.load('it_core_news_sm')

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
    if prob.item() > 0.50:
        if tag == "orario":
            return "Oggi è: " + str(datetime.now())
        if tag == "search_game":
            extract_game_and_price(msg)
            return "Cerca il gioco nel database..."
        else:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    return random.choice(intent['responses'])
    
    return "Non ho capito, potresti riprovare ?"

    # Funzione per estrarre il tipo di gioco e il prezzo
def extract_game_and_price(msg):
    # Esegui il parsing della frase con SpaCy
    doc = nlp(msg)
    
    # Trova il tipo di gioco
    game_types = ['avventura', 'strategia', 'simulazione', 'ruolo', 'azione', 'sportivi', 'corse', 'combattimento', 'horror', 'platform']
    game_found = None
    for token in doc:
        if token.text.lower() in game_types:
            game_found = token.text.lower()
            break
    
    # Trova il prezzo usando espressioni regolari
    price_match = re.search(r"(meno di|sotto i?|a meno di?|più di?|sopra di?|di?)\s*(\d+)", msg)
    price_found = None
    if price_match:
        price_found = int(price_match.group(2))  # Estrai il numero del prezzo
    
    print(f"Query: {msg}")
    print(f"Tipo di gioco: {game_found}")
    print(f"Tipe search : {price_match.group(1)}")
    print(f"Prezzo richiesto: {price_found} euro")
    print("-" * 50)
    
    return game_found, price_found