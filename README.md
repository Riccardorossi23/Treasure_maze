# 🗺️ Treasure Maze

**Un labirinto che si legge da una foto.**
Una CNN riconosce muri, tesori e percorsi da un'immagine, e due algoritmi di ricerca (A* e UCS) trovano il tragitto migliore per raccoglierli.

Progetto realizzato per il corso di **Intelligenza Artificiale** — Università degli Studi di Perugia, Dipartimento di Matematica e Informatica.

---

## 💡 L'idea

Immagina un labirinto disegnato o generato come semplice immagine: alcune celle sono muri, altre nascondono un tesoro, altre ancora sono più "costose" da attraversare. Invece di codificare a mano la mappa, il progetto la **legge direttamente dall'immagine** tramite una rete neurale convoluzionale, la trasforma in un problema di ricerca e lo risolve trovando il percorso ottimale per raccogliere i tesori — anche abbattendo qualche muro, se conviene.

## ⚙️ Come funziona

La pipeline è composta da quattro passaggi:

1. **Generazione del labirinto** — un labirinto può essere generato casualmente (in Python o direttamente nel browser) oppure disegnato a mano, esportandolo come immagine.
2. **Lettura dell'immagine (CNN)** — l'immagine viene divisa in celle, e una piccola rete convoluzionale classifica ciascuna cella come punto di partenza, tesoro, muro o cella libera (con relativo costo di attraversamento).
3. **Costruzione del problema** — la griglia ottenuta viene modellata come un problema di ricerca: lo stato tiene traccia della posizione dell'agente, dei tesori già raccolti e dei muri già abbattuti.
4. **Ricerca del percorso** — il problema viene risolto sia con **A*** (guidato da un'euristica sulla distanza dal tesoro più vicino) sia con **UCS**, confrontandone tempi e prestazioni. Il percorso trovato viene infine disegnato sopra l'immagine originale.

## 🧰 Tecnologie utilizzate

- **Python**
- **TensorFlow / Keras** — rete convoluzionale per la classificazione delle celle
- **OpenCV / Pillow** — elaborazione delle immagini
- **AIMA search** — libreria di algoritmi di ricerca (A*, UCS) fornita dal corso

## 📁 Struttura del repository

```
Treasure_Maze/
├── main.py                     CLI per addestrare il modello e risolvere un labirinto
├── Treasure_Maze_fixed.ipynb   stessa pipeline in versione notebook (Colab/Jupyter)
├── treasure_maze/
│   ├── maze_problem.py         definizione del problema di ricerca (stato, costi, euristica)
│   ├── classifier.py           addestramento e uso della CNN
│   ├── visualize.py            disegna il percorso trovato sull'immagine
│   └── generate_test_maze.py   genera labirinti casuali di prova
│
generazione_labirinto/
├── index.html                  generatore di labirinti nel browser
└── generazione_mappa.py        stesso generatore in Python

Progetto AI.pdf                 relazione completa del progetto
```

## ▶️ Come provarlo

**La strada più semplice: Google Colab.**
Basta aprire `Treasure_Maze_fixed.ipynb` su [Google Colab](https://colab.research.google.com/), caricare i file di supporto richiesti (indicati nella prima cella) ed eseguire le celle in ordine: al momento giusto, il notebook chiederà l'immagine del labirinto da risolvere.

**In locale**, con Python 3.10–3.13:

```bash
git clone https://github.com/Riccardorossi23/Treasure_maze.git
cd Treasure_maze/Treasure_Maze

pip install tensorflow opencv-python scikit-learn pillow
unzip aima.zip
unzip maze_data.zip

python main.py train --data maze_data          # addestra il classificatore
python main.py solve --image labirinto.png     # risolve un labirinto, raccogliendo tutti i tesori
python main.py solve --image labirinto.png --k 2   # ...o almeno 2 di essi
```

## 🔧 Dal prototipo alla versione funzionante

La prima versione del notebook, scritta pensando solo all'ambiente Colab, conteneva diversi problemi strutturali che ho affrontato in una fase successiva di refactoring: classi duplicate e mai completate, codice morto, un bug nel conteggio degli stati esplorati che rendeva incoerente il confronto tra A* e UCS, muri trattati come invalicabili invece che abbattibili, e una gestione della ricerca multi-tesoro che non permetteva di fermarsi dopo averne raccolti "almeno k".

Il risultato è una versione più solida, eseguibile anche fuori da Colab, con una singola definizione pulita del problema e un conteggio corretto delle prestazioni degli algoritmi — un buon esercizio di debug e refactoring oltre che di intelligenza artificiale.

## 🧩 Il problema in breve

- **Celle libere**: costo di transito da 1 a 4, secondo il numero indicato.
- **Muri**: non attraversabili direttamente, ma abbattibili a un costo fisso; una volta abbattuti diventano permanentemente percorribili.
- **Tesori**: raccolti semplicemente passandoci sopra; l'obiettivo è configurabile — raccoglierli tutti, oppure solo un numero minimo.
- **Movimento**: su, giù, sinistra, destra, senza uscire dalla griglia.

I dettagli formali e i risultati sperimentali su diversi esempi di labirinto sono nella relazione `Progetto AI.pdf`.

---

## 👤 Autore

**Riccardo Rossi**
Progetto sviluppato per il corso di Intelligenza Artificiale, Università degli Studi di Perugia.
