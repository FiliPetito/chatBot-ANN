import numpy as np
import nltk

"""Utilizzo di Spacy per un migliore supporto per la tokenizzazione in italiano"""
import spacy
nlp = spacy.load("it_core_news_sm")

nltk.download('all')
#from nltk.stem.porter import PorterStemmer

"""SnowballStemmer di NLTK per l'ottimizzazione in italiano"""
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("italian")  # Imposta lo stemmer sulla lingua italiana

def tokenize(sentence):
    doc = nlp(sentence)
    return [token.text for token in doc]


def stem(word):
    """
    stemming = find the root form of the word
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag