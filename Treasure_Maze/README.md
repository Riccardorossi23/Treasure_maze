# Treasure Maze — versione corretta

Questa cartella contiene la stessa idea progettuale del notebook originale
(`Treasure_Maze.ipynb`), ma sistemata perché **non era funzionante**. Qui
sotto l'elenco dei problemi trovati e come sono stati risolti.

## Problemi trovati nell'originale

1. **Quattro definizioni della classe `TreasureMaze`** nello stesso file,
   di cui due erano scheletri con metodi `# Placeholder` mai completati.
   In Python l'ultima definizione sovrascrive le precedenti, quindi il
   codice "funzionava per caso": bastava eseguire le celle in un ordine
   diverso, o cancellarne una, per rompere tutto silenziosamente.
2. Funzioni mai utilizzate (`best_first_search_graph`, `ucs`, `astar`,
   `misplaced_tiles`, `manh`) che usavano `PriorityQueue` e `memoize`
   **senza importarli**: sarebbero andate in errore se richiamate.
3. Comandi Colab (`!pip install`, `!apt-get`, `files.upload()`,
   `drive.mount()`) mescolati alla logica applicativa: il notebook non
   poteva essere eseguito né come script, né in locale, né due volte di
   fila senza ripetere upload manuali.
4. **Bug nel conteggio delle "celle esplorate"** (capitolo 3 del pdf): il
   codice usava la lunghezza del *percorso soluzione* al posto del numero
   di stati realmente generati durante la ricerca. Per questo A* e UCS
   risultavano — sempre, in ogni esempio — con lo stesso numero di celle
   esplorate, un risultato incoerente con la teoria (UCS, senza euristica,
   esplora tipicamente più stati di A*).
5. I muri (`X`) erano trattati come celle **invalicabili**, mentre la
   consegna li descrive esplicitamente come *abbattibili a costo 5* (quindi
   attraversabili, pagando un prezzo).
6. La ricerca del percorso lanciava una ricerca A*/UCS **separata per ogni
   tesoro**, visitandoli nell'ordine in cui comparivano nella lista. Questo
   non è un vero problema di ricerca multi-obiettivo, non garantisce un
   percorso complessivo ottimale e **non permette affatto** la modalità
   "raccogli almeno k tesori" richiesta dalla consegna (capitolo 1).
7. `extract_maze` apriva una finestra `plt.show()` per **ogni singola
   cella** dell'immagine (64 finestre per un labirinto 8×8) e sovrascriveva
   la predizione del modello con soglie di probabilità arbitrarie
   (`prob_S > 0.05`), invece di usare semplicemente la classe più probabile.
8. `generazione_mappa.py` (generatore di labirinti di test) non garantiva
   **mai** la presenza di almeno un tesoro nella griglia generata, e
   apriva l'immagine risultante con un comando valido solo su Windows.

## Struttura

```
treasure_maze/
  maze_problem.py        problema di ricerca (A*/UCS), un'unica classe corretta
  classifier.py           CNN: training, salvataggio/caricamento pesi, estrazione da immagine
  visualize.py             disegna il percorso trovato sull'immagine
  generate_test_maze.py    genera labirinti casuali di prova (bug corretti)
main.py                    CLI: train / solve
Treasure_Maze_fixed.ipynb  stesso codice, in un notebook Colab pulito e ordinato
aima.zip, maze_data.zip    invariati rispetto all'originale
```

## Uso da terminale

```bash
pip install tensorflow opencv-python scikit-learn pillow

# addestra il classificatore (una volta sola: i pesi vengono salvati su disco)
unzip aima.zip && unzip maze_data.zip
python main.py train --data maze_data

# risolvi un labirinto da un'immagine
python main.py solve --image labirinto.png            # tutti i tesori
python main.py solve --image labirinto.png --k 2       # almeno 2 tesori
```

## Uso da Colab / Jupyter

Apri `Treasure_Maze_fixed.ipynb` ed esegui le celle in ordine: la logica
di training/estrazione/risoluzione è identica a quella dei moduli sopra,
solo commentata passo per passo per la relazione.

## Cosa NON è cambiato

La formalizzazione del dominio (capitolo 1.1 del pdf), l'architettura CNN
(convoluzione → pooling → dense, capitolo 2.2) e l'uso di A*/UCS di AIMA
(capitolo 2.5) sono la stessa idea del progetto originale: è stata sistemata
l'implementazione, non il progetto.
