import pkg_resources
import subprocess
import sys

# Librerie richieste
required = {'spacy'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

# Installa le librerie mancanti e il modello di lingua per l'italiano
if missing:
    # Aggiorna pip
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
    # Installa le librerie mancanti
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])

"""Codice por l'auto installazione della dipendeza"""
try:
    import spacy
    subprocess.check_call([sys.executable, '-m', 'spacy', 'download', 'it_core_news_sm'])
    print("Il modello di lingua 'it_core_news_sm' è stato installato correttamente.")
except ImportError:
    print("Errore: spaCy non è installato correttamente. 2")
