import numpy as np
import nltk
from nltk.corpus import stopwords
"""Utilizzo di Spacy per un migliore supporto per la tokenizzazione in italiano"""
import spacy
from spellchecker import SpellChecker

# Lista di parole imprecise o inutili da rimuovere
FILTERED_WORDS = {'Arrivedere ci', 'dire mi', 'papa', 'pare li', 'pi'}

# Carica il modello Spacy per l'italiano
nlp = spacy.load("it_core_news_sm")
nltk.download('stopwords')

spell = SpellChecker(language='it')
# Definisci le stopwords
STOP_WORDS = set(stopwords.words('italian'))

# Funzione di correzione ortografica
def correct_spelling(words):
    corrected_words = []
    for word in words:
        corrected_word = spell.correction(word)  # Correggi la parola
        corrected_words.append(corrected_word if corrected_word else word)
    return corrected_words

def lemmatize(text):
    # Usa spaCy per la lemmatizzazione
    doc = nlp(text)
    # Filtra i lemmi per escludere stopwords e parole filtrate
    tokens = [token.lemma_ for token in doc if token.lemma_ not in STOP_WORDS and token.lemma_ not in FILTERED_WORDS and not token.is_punct]
    return tokens

# Funzione per tokenizzare e correggere l'ortografia di una frase
def tokenize_and_correct(sentence):
    doc = nlp(sentence)
    tokens = [token.text for token in doc]
    tokens = correct_spelling(tokens)  # Applica la correzione ortografica
    # Rimuove parole filtrate e stop words
    tokens = [token for token in tokens if token not in FILTERED_WORDS and token not in STOP_WORDS]
    return tokens

def bag_of_words(tokenized_sentence, words):
    # Usa la lemmatizzazione e appiattisce la lista di token
    sentence_words = [lemmatize(word)[0] for word in tokenized_sentence if lemmatize(word)]
    sentence_words = correct_spelling(sentence_words) 
    sentence_words = set(sentence_words)  # Riduce i duplicati per migliorare la ricerca
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1
    return bag