# ChatBot-ANN
ChatBot basato su rete neurale artificiale (ANN). Il bot è progettato per riconoscere le intezioni dell'utente basate su input e rispondere in modo appropriato.

## Intent JSON File
Il file intents.json definisce la base di conoscenza del bot. È strutturato in una lista di intezioni, ciascuna con:
  - Tag: Un identificatore unico per l'intezione.
  - Patterns: Frasi che rappresentano esempi di input dell'utente.
  - Response: Risposte predefinite che il bot può utilizzare.
Funzionamento
  1. Le frasi nei patterns vengono usate per addestrare il modello.
  2. Durante l'inferenza, il bot confronta l'input dell'utente con queste frasi, cercando la probabilità di appartenenza a ciascun tag.
