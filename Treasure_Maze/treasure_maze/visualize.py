# -*- coding: utf-8 -*-
"""
Visualizzazione del percorso trovato sull'immagine originale del
labirinto.

Rispetto all'originale, `visualize_solution` ora salva sempre l'immagine
su disco (oltre a poterla mostrare), perché uno script non interattivo
(o eseguito da terminale/CI) non può fare affidamento su `plt.show()`
per produrre un output utilizzabile.
"""
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

CELL_SIZE = 28
Position = Tuple[int, int]


def visualize_solution(
    image_path: str,
    path: List[Position],
    output_path: str = "maze_solution.png",
    show: bool = False,
    cell_size: int = CELL_SIZE,
) -> str:
    """Disegna il percorso `path` (lista di (riga, colonna)) sull'immagine
    del labirinto in `image_path`, evidenziando le celle attraversate."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Immagine non trovata o non valida: {image_path}")

    overlay = img.copy()
    for row, col in path:
        y0, x0 = row * cell_size, col * cell_size
        cv2.rectangle(overlay, (x0, y0), (x0 + cell_size, y0 + cell_size), (0, 0, 255), -1)

    blended = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
    cv2.imwrite(output_path, blended)

    if show:
        import matplotlib.pyplot as plt

        plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
        plt.title("Labirinto con percorso")
        plt.axis("off")
        plt.show()

    return output_path
