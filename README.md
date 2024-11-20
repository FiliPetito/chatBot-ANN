# ChatBot-ANN
ChatBot basato su rete neurale artificiale (ANN). Il bot è progettato per riconoscere le intezioni dell'utente basate su input e rispondere in modo appropriato.

## Intent JSON File
Il file intents.json definisce la base di conoscenza del bot. È strutturato in una lista di intezioni, ciascuna con:
  - **Tag**: Un identificatore unico per l'intezione.
  - **Patterns**: Frasi che rappresentano esempi di input dell'utente.
  - **Response**: Risposte predefinite che il bot può utilizzare.
### Funzionamento
  1. Le frasi nei patterns vengono usate per addestrare il modello.
  2. Durante l'inferenza, il bot confronta l'input dell'utente con queste frasi, cercando la probabilità di appartenenza a ciascun tag.

## Pre-elaborazione dei dati
<img src="ImageRif/preElaborazioneDati.png" style="width: 50%"/>

### Descrizione 
1. Obiettivo: Creare un dataset strutturato per addestrare il modello.
2. Processo:
   -  **Tokenizzazione**: ogni frase in patterns viene suddivisa in parole (tokens).
   -  **Lista parole totali**: tutte le parole vengono aggiunte a all_words.
   -  **Coppie parola-tag**: ogni coppia (parole, tag) viene salvata in xy.

## Filtraggio e Normalizzazione
<img src="ImageRif/preElaborazioneDati.png" style="width: 50%"/>

