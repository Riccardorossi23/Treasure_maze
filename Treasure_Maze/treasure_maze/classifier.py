# -*- coding: utf-8 -*-
"""
Classificazione delle celle del labirinto (CNN) ed estrazione della
matrice del labirinto da un'immagine.

Differenze principali rispetto all'originale:
- Nessuna dipendenza da Google Colab (niente `google.colab.files`,
  `drive.mount`, comandi `!pip`/`!apt`): il codice originale non poteva
  proprio essere eseguito al di fuori di un notebook Colab, mentre questo
  modulo funziona come normale script/libreria Python.
- I pesi del modello vengono salvati e ricaricati (`load_or_train_model`),
  quindi non serve riaddestrare la CNN a ogni esecuzione.
- `extract_maze` non stampa più decine di `plt.show()` per ogni singola
  cella (nell'originale, per un labirinto 8x8 venivano aperte 64 finestre
  con `plt.imshow`): questo rendeva lo script inutilizzabile in modo
  automatico/non interattivo. Ora il debug visivo è opzionale
  (`debug=True`) e disegna una griglia unica invece di un plot per cella.
- Le soglie di probabilità "magiche" (`prob_S > 0.05`, `prob_T > 0.05`)
  che nell'originale sovrascrivevano la predizione del modello quasi a
  caso sono state rimosse: si usa semplicemente la classe con
  probabilità massima (`argmax`), che è già la predizione della rete.
- Viene validato che la matrice estratta contenga esattamente una 'S' e
  almeno una 'T', con un errore chiaro se non è così (il progetto
  richiede esplicitamente questo vincolo, capitolo 4.1 del pdf).
"""
from __future__ import annotations

import os
from typing import List, Tuple

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

CELL_SIZE = 28
WEIGHTS_PATH = "maze_classifier.weights.h5"
CLASSES_PATH = "maze_classifier.classes.npy"


def _lazy_import_tf():
    """Importa tensorflow solo quando serve, per non rendere l'intero
    pacchetto inutilizzabile (es. per chi vuole solo la parte di ricerca)
    se tensorflow non è installato."""
    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError as exc:
        raise ImportError(
            "tensorflow non è installato. Esegui: pip install tensorflow"
        ) from exc
    return Sequential, Conv2D, MaxPooling2D, Flatten, Dense, Input, Adam, EarlyStopping


def load_training_data(data_path: str):
    """Carica le immagini 28x28 di training, organizzate in
    sottocartelle: una per ciascuna classe ('S', 'T', 'X', '1'..'4')."""
    images, labels = [], []
    classes = sorted(
        d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))
    )
    if not classes:
        raise ValueError(f"Nessuna sottocartella di classe trovata in '{data_path}'.")

    for label in classes:
        class_path = os.path.join(data_path, label)
        for img_name in os.listdir(class_path):
            img = cv2.imread(os.path.join(class_path, img_name), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (CELL_SIZE, CELL_SIZE)) / 255.0
            images.append(img)
            labels.append(label)

    if not images:
        raise ValueError(f"Nessuna immagine valida trovata in '{data_path}'.")

    label_binarizer = LabelBinarizer()
    label_binarizer.fit(classes)

    images = np.array(images).reshape(-1, CELL_SIZE, CELL_SIZE, 1)
    labels = label_binarizer.transform(labels)
    return images, labels, label_binarizer


def build_model(num_classes: int):
    Sequential, Conv2D, MaxPooling2D, Flatten, Dense, Input, Adam, _ = _lazy_import_tf()
    model = Sequential([
        Input(shape=(CELL_SIZE, CELL_SIZE, 1)),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train_model(data_path: str, epochs: int = 14, batch_size: int = 32):
    """Addestra la CNN sulle immagini di training e salva pesi + classi
    su disco, così le esecuzioni successive possono limitarsi a
    ricaricarli con `load_or_train_model`."""
    _, _, _, _, _, _, _, EarlyStopping = _lazy_import_tf()

    images, labels, label_binarizer = load_training_data(data_path)
    train_images, val_images, train_labels, val_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )

    model = build_model(num_classes=len(label_binarizer.classes_))
    early_stopping = EarlyStopping(monitor="val_loss", patience=5)
    model.fit(
        train_images, train_labels,
        epochs=epochs, batch_size=batch_size,
        validation_data=(val_images, val_labels),
        callbacks=[early_stopping],
    )

    model.save_weights(WEIGHTS_PATH)
    np.save(CLASSES_PATH, label_binarizer.classes_)
    return model, label_binarizer


def load_or_train_model(data_path: str, force_retrain: bool = False, epochs: int = 14):
    """Ricarica il modello già addestrato se presente su disco, altrimenti
    lo addestra da zero (comportamento assente nell'originale, dove ogni
    esecuzione riaddestrava sempre tutto)."""
    if not force_retrain and os.path.exists(WEIGHTS_PATH) and os.path.exists(CLASSES_PATH):
        from sklearn.preprocessing import LabelBinarizer as _LB

        classes = np.load(CLASSES_PATH, allow_pickle=True)
        label_binarizer = _LB()
        label_binarizer.fit(classes)
        model = build_model(num_classes=len(classes))
        model.load_weights(WEIGHTS_PATH)
        return model, label_binarizer

    return train_model(data_path, epochs=epochs)


def extract_maze(image_path: str, model, label_binarizer, debug: bool = False) -> List[List[str]]:
    """Suddivide l'immagine del labirinto in celle 28x28 e classifica
    ciascuna cella con il modello addestrato."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Immagine non trovata o non valida: {image_path}")

    img_height, img_width = img.shape
    if img_height % CELL_SIZE != 0 or img_width % CELL_SIZE != 0:
        raise ValueError(
            f"L'immagine {img_width}x{img_height} non ha dimensioni multiple di "
            f"{CELL_SIZE}px: rigenerala con lo stesso cell_size usato in training."
        )

    num_rows, num_cols = img_height // CELL_SIZE, img_width // CELL_SIZE
    maze_matrix: List[List[str]] = [["" for _ in range(num_cols)] for _ in range(num_rows)]

    for i in range(num_rows):
        for j in range(num_cols):
            y0, x0 = i * CELL_SIZE, j * CELL_SIZE
            cell = img[y0:y0 + CELL_SIZE, x0:x0 + CELL_SIZE]
            cell_input = (cell / 255.0).reshape(1, CELL_SIZE, CELL_SIZE, 1)

            prediction = model.predict(cell_input, verbose=0)
            predicted_label = label_binarizer.classes_[int(np.argmax(prediction, axis=1)[0])]
            maze_matrix[i][j] = str(predicted_label)

    _validate_maze(maze_matrix)

    if debug:
        _draw_debug_grid(img, maze_matrix)

    return maze_matrix


def _validate_maze(maze: List[List[str]]) -> None:
    flat = [cell for row in maze for cell in row]
    n_start = flat.count("S")
    n_treasure = flat.count("T")
    if n_start != 1:
        raise ValueError(
            f"Il labirinto estratto deve contenere esattamente una 'S' (trovate {n_start})."
        )
    if n_treasure < 1:
        raise ValueError("Il labirinto estratto deve contenere almeno un tesoro 'T'.")


def _draw_debug_grid(img: np.ndarray, maze: List[List[str]]) -> None:
    """Disegna in un'unica figura l'esito della classificazione, al posto
    delle decine di finestre separate aperte dall'originale."""
    import matplotlib.pyplot as plt

    rows, cols = len(maze), len(maze[0])
    fig, ax = plt.subplots(figsize=(cols, rows))
    ax.imshow(img, cmap="gray")
    for i in range(rows):
        for j in range(cols):
            ax.text(
                j * CELL_SIZE + CELL_SIZE / 2, i * CELL_SIZE + CELL_SIZE / 2,
                maze[i][j], color="red", ha="center", va="center", fontsize=10, weight="bold",
            )
    ax.set_title("Classificazione celle")
    ax.axis("off")
    plt.show()
