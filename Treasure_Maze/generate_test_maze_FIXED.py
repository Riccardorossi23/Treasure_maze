# -*- coding: utf-8 -*-
"""
Generatore di immagini di labirinti casuali, utile per testare
`extract_maze` e la pipeline di risoluzione senza dover disegnare a mano
un'immagine.

Bug corretti rispetto all'originale (`generazione_mappa.py`):
- il labirinto generato non conteneva MAI la garanzia di avere almeno un
  tesoro 'T': 'T' era solo una delle 6 possibilità scelte a caso per ogni
  cella di riempimento, quindi capitava spesso di generare labirinti privi
  di tesori, non risolvibili (il progetto richiede sempre almeno un
  tesoro, capitolo 4.1 del pdf).
- l'apertura automatica del file (`os.system("start ...")`) funzionava
  solo su Windows; ora viene rilevata la piattaforma.
"""
from __future__ import annotations

import platform
import random
import subprocess
from typing import List

from PIL import Image, ImageDraw, ImageFont

CELL_SIZE = 28
SYMBOLS = ["1", "2", "3", "4", "X"]


def generate_random_maze(size: int, min_treasures: int = 1) -> List[List[str]]:
    if size < 2:
        raise ValueError("size deve essere almeno 2 (serve spazio per S e almeno un T).")
    max_treasures = max(min_treasures, size - 1)

    maze = [["" for _ in range(size)] for _ in range(size)]
    cells = [(r, c) for r in range(size) for c in range(size)]
    random.shuffle(cells)

    start_cell = cells.pop()
    maze[start_cell[0]][start_cell[1]] = "S"

    n_treasures = random.randint(min_treasures, max_treasures)
    for _ in range(n_treasures):
        r, c = cells.pop()
        maze[r][c] = "T"

    for r, c in cells:
        maze[r][c] = random.choice(SYMBOLS)

    return maze


def generate_maze_image(maze: List[List[str]], cell_size: int = CELL_SIZE, font_size: int = 20) -> Image.Image:
    rows, cols = len(maze), len(maze[0])
    img = Image.new("L", (cols * cell_size, rows * cell_size), color=255)
    draw = ImageDraw.Draw(img)

    for i in range(rows + 1):
        draw.line((0, i * cell_size, cols * cell_size, i * cell_size), fill=0, width=2)
    for j in range(cols + 1):
        draw.line((j * cell_size, 0, j * cell_size, rows * cell_size), fill=0, width=2)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    for i in range(rows):
        for j in range(cols):
            x, y = j * cell_size + cell_size // 2, i * cell_size + cell_size // 2
            draw.text((x, y), maze[i][j], fill=0, font=font, anchor="mm")

    return img


def save_random_maze_image(size: int, output_path: str = "maze_image.png", open_file: bool = False) -> str:
    maze = generate_random_maze(size)
    img = generate_maze_image(maze)
    img.save(output_path)

    if open_file:
        system = platform.system()
        opener = {"Windows": ["start", output_path], "Darwin": ["open", output_path]}.get(
            system, ["xdg-open", output_path]
        )
        try:
            subprocess.run(opener, shell=(system == "Windows"), check=False)
        except OSError:
            pass

    return output_path


if __name__ == "__main__":
    import sys

    size = int(sys.argv[1]) if len(sys.argv) > 1 else random.randint(4, 8)
    path = save_random_maze_image(size)
    print(f"Labirinto {size}x{size} salvato in: {path}")
