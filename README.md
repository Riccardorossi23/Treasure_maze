# Treasure Maze

Progetto per il corso di **Intelligenza Artificiale** — Università degli Studi di Perugia, Dipartimento di Matematica e Informatica, A.A. 2024-2025.

**Autore:** Dott. Rossi Riccardo

Treasure Maze è una variante del labirinto classico in cui l'agente deve raccogliere dei tesori. Il labirinto viene fornito come **immagine**: un modello di rete neurale convoluzionale (CNN) classifica ogni cella dell'immagine (partenza, tesoro, muro, costo), ricostruisce la griglia e la risolve con algoritmi di ricerca A* e UCS (Uniform Cost Search), tenendo conto dei costi di attraversamento e della possibilità di abbattere i muri.

---

## Indice

- [Come funziona](#come-funziona)
- [Struttura del repository](#struttura-del-repository)
- [Come eseguirlo](#come-eseguirlo)
  - [Su Google Colab](#su-google-colab)
  - [In locale](#in-locale)
- [Storia del progetto: dalla versione rotta a quella funzionante](#storia-del-progetto-dalla-versione-rotta-a-quella-funzionante)
- [Dominio del problema](#dominio-del-problema)

---

## Come funziona

La pipeline è composta da quattro fasi:

1. **Generazione/preparazione dell'immagine del labirinto** (`generazione_labirinto/`): un labirinto può essere generato casualmente (pagina web `index.html`, oppure script Python `generazione_mappa.py`), oppure disegnato/fotografato a mano, purché ogni cella occupi un blocco di 28×28 pixel — la stessa dimensione usata per addestrare il classificatore.

2. **Classificazione delle celle (CNN)**: l'immagine viene divisa in celle 28×28 e ciascuna viene classificata da una piccola rete convoluzionale (convoluzione → pooling → strato denso) in una delle categorie:
   - `S` — posizione di partenza dell'agente
   - `T` — tesoro
   - `X` — muro, abbattibile a costo 5
   - `1`/`2`/`3`/`4` — cella libera, con il relativo costo di attraversamento

3. **Rappresentazione come problema di ricerca**: la matrice ottenuta viene modellata come un problema AIMA (`Problem`), il cui stato include la posizione dell'agente, i tesori già raccolti e i muri già abbattuti. Questo permette di risolvere sia l'obiettivo "raccogli **tutti** i tesori" sia "raccoglierne **almeno k**".

4. **Ricerca del percorso**: il problema viene risolto con **A\*** (euristica: distanza di Manhattan dal tesoro non raccolto più vicino) e con **UCS**, misurando tempo di esecuzione e numero di stati effettivamente esplorati, per confrontare le prestazioni dei due algoritmi. Il percorso trovato viene infine disegnato sull'immagine originale.

## Struttura del repository

```
Treasure_Maze/
├── main.py                        CLI: addestramento e risoluzione da terminale
├── Treasure_Maze_fixed.ipynb      stessa pipeline, in notebook Colab/Jupyter
├── aima.zip                       libreria di ricerca AIMA (fornita dal corso)
├── maze_data.zip                  immagini di training per il classificatore (28x28, una cartella per classe)
├── README.md                      questo file
└── treasure_maze/
    ├── maze_problem.py            problema di ricerca (stato, azioni, costo, euristica, A*/UCS)
    ├── classifier.py              training CNN, salvataggio/caricamento pesi, estrazione labirinto da immagine
    ├── visualize.py                disegna il percorso trovato sull'immagine
    └── generate_test_maze.py       genera labirinti casuali di prova

generazione_labirinto/
├── index.html                     generatore di labirinti nel browser (esportabile in PNG)
└── generazione_mappa.py           stesso generatore, versione Python

Progetto AI.pdf                    relazione del progetto
```

## Come eseguirlo

### Su Google Colab

1. Apri [Google Colab](https://colab.research.google.com/) e carica `Treasure_Maze/Treasure_Maze_fixed.ipynb` (**File → Carica blocco note**).
2. Nella barra laterale (icona a forma di cartella) carica `aima.zip` e `maze_data.zip` nella cartella principale della sessione (`/content/`).
   > Questi file spariscono alla chiusura della sessione: se vuoi evitare di ricaricarli ogni volta, monta Google Drive e copiali lì (istruzioni nella prima cella del notebook).
3. Esegui le celle in ordine (**Runtime → Esegui tutto**, oppure `Shift+Invio` cella per cella).
4. Nella sezione "Estrazione e risoluzione di un labirinto da immagine" ti verrà chiesto di caricare l'immagine del labirinto da risolvere.

### In locale

Richiede Python 3.10–3.13 (TensorFlow non supporta ancora Python 3.14).

```bash
git clone https://github.com/Riccardorossi23/Treasure_maze.git
cd Treasure_maze/Treasure_Maze

pip install tensorflow opencv-python scikit-learn pillow

# estrai le librerie/dati (una volta sola)
unzip aima.zip
unzip maze_data.zip

# addestra il classificatore (i pesi vengono salvati su disco)
python main.py train --data maze_data

# risolvi un labirinto da immagine
python main.py solve --image labirinto.png            # raccogliere tutti i tesori
python main.py solve --image labirinto.png --k 2       # raccoglierne almeno 2
```

Su Windows, se `python`/`pip` non vengono riconosciuti, usa il launcher `py`:
```powershell
py -m pip install tensorflow opencv-python scikit-learn pillow
py main.py train --data maze_data
```

## Storia del progetto: dalla versione rotta a quella funzionante

La prima versione del notebook (`Treasure_Maze.ipynb`, conservata nella cronologia del repository) non era funzionante fuori da un ambiente Google Colab e conteneva diversi bug strutturali, corretti in questa versione:

| Problema originale | Correzione |
|---|---|
| **4 definizioni diverse** della classe `TreasureMaze` nello stesso file (due erano scheletri con metodi placeholder mai completati) | Un'unica classe `TreasureMaze`, ben definita, in `treasure_maze/maze_problem.py` |
| Funzioni mai usate (`best_first_search_graph`, `ucs`, `astar`, `misplaced_tiles`, `manh`) che referenziavano `PriorityQueue`/`memoize` senza importarli | Rimosse (codice morto) |
| Comandi Colab (`!pip`, `!apt`, `files.upload()`, `drive.mount()`) mescolati alla logica applicativa | Isolati e resi opzionali con un controllo `IN_COLAB`; il progetto ora gira anche come script locale |
| **Bug nel conteggio delle "celle esplorate"** (capitolo 3 della relazione): veniva usata la lunghezza del *percorso soluzione* invece del numero di stati realmente generati durante la ricerca — per questo A* e UCS risultavano sempre con lo stesso numero di celle esplorate, un dato incoerente con la teoria | Conteggio corretto tramite `aima.search.InstrumentedProblem`, che misura gli stati effettivamente generati |
| I muri (`X`) erano trattati come celle **invalicabili** | Ora sono attraversabili abbattendoli a costo 5, come richiesto dalla consegna |
| La ricerca lanciava un A*/UCS **separato per ogni tesoro**, in ordine di lista — non un vero problema multi-obiettivo, e non permetteva affatto la modalità "almeno k tesori" | Lo stato del problema include ora i tesori raccolti, così una singola ricerca risolve sia "tutti i tesori" sia "almeno k" |
| `extract_maze` apriva **una finestra per ogni cella** dell'immagine (64 popup per un labirinto 8×8) e sovrascriveva le predizioni della CNN con soglie di probabilità arbitrarie (`prob_S > 0.05`) | Debug visivo opzionale con un'unica griglia; si usa semplicemente la classe più probabile predetta dal modello |
| Il generatore di labirinti di prova non garantiva **mai** la presenza di almeno un tesoro | Garantito almeno un tesoro nella griglia generata |
| Apertura automatica del file immagine funzionante solo su Windows (`os.system("start ...")`) | Rilevamento automatico della piattaforma (Windows/macOS/Linux) |

## Dominio del problema

- **Celle calpestabili**: costo di transito 1-4, indicato dal numero nella cella.
- **Muri (`X`)**: non attraversabili direttamente; possono essere abbattuti a un costo fisso di 5, dopodiché diventano permanentemente attraversabili a costo 1.
- **Partenza (`S`)**: posizione iniziale dell'agente (una sola per labirinto).
- **Tesori (`T`)**: l'agente li raccoglie semplicemente entrando nella cella. Obiettivo configurabile: raccoglierli tutti, oppure almeno `k`.
- **Azioni**: spostamento in una delle 4 celle adiacenti (su/giù/sinistra/destra); l'agente non può uscire dalla griglia.
- **Percorso ottimale**: calcolato minimizzando il costo totale del tragitto (A* e UCS restituiscono lo stesso costo ottimo; A* è generalmente più rapido grazie all'euristica).

Per i dettagli formali e i risultati sperimentali su tre esempi di labirinto (dimensioni 6×6, 4×4, 8×8), vedi `Progetto AI.pdf`.
