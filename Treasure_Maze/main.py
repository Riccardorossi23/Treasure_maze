# -*- coding: utf-8 -*-
"""
Punto di ingresso del progetto Treasure Maze.

Uso:
    python main.py train --data maze_data [--epochs 14]
        Addestra (o riaddestra) la CNN sulle immagini in `maze_data/` e
        salva i pesi su disco.

    python main.py solve --image labirinto.png [--k 2] [--debug]
        Carica il modello (addestrandolo se non è ancora presente),
        estrae il labirinto dall'immagine, lo risolve con A* e UCS e
        salva `maze_solution.png` con il percorso evidenziato,
        stampando anche il confronto di prestazioni (capitolo 3 del pdf).

Questo file sostituisce l'unico, enorme notebook Colab originale, in cui
addestramento, generazione dati e risoluzione erano tutti mescolati in
un'unica sequenza di celle non eseguibile fuori da Colab (comandi `!pip`,
`!apt`, `files.upload()`, `drive.mount()` sparsi nel mezzo della logica).
"""
from __future__ import annotations

import argparse
import sys

from treasure_maze.classifier import extract_maze, load_or_train_model, train_model
from treasure_maze.maze_problem import find_start, solve_treasure_maze
from treasure_maze.visualize import visualize_solution


def cmd_train(args: argparse.Namespace) -> None:
    train_model(args.data, epochs=args.epochs)
    print(f"Modello addestrato e salvato in 'maze_classifier.weights.h5'.")


def cmd_solve(args: argparse.Namespace) -> None:
    model, label_binarizer = load_or_train_model(args.data, epochs=args.epochs)
    maze = extract_maze(args.image, model, label_binarizer, debug=args.debug)

    print("Matrice del labirinto estratta:")
    for row in maze:
        print(" ".join(row))

    start = find_start(maze)
    astar_result, ucs_result = solve_treasure_maze(maze, start=start, k=args.k)

    for result in (astar_result, ucs_result):
        print(f"\n--- {result.algorithm.upper()} ---")
        if result.path is None:
            print("Nessuna soluzione trovata.")
            continue
        print(f"Tempo di esecuzione: {result.time_seconds:.4f} secondi")
        print(f"Costo del percorso: {result.cost}")
        print(f"Celle esplorate (stati generati): {result.states_generated}")
        print(f"Percorso: {result.path}")

    if astar_result.path:
        out = visualize_solution(args.image, astar_result.path, show=args.debug)
        print(f"\nImmagine con il percorso (A*) salvata in: {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Treasure Maze")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Addestra la CNN di classificazione celle")
    train_p.add_argument("--data", default="maze_data", help="Cartella con le immagini di training")
    train_p.add_argument("--epochs", type=int, default=14)
    train_p.set_defaults(func=cmd_train)

    solve_p = sub.add_parser("solve", help="Risolve un labirinto a partire da un'immagine")
    solve_p.add_argument("--image", required=True, help="Percorso dell'immagine del labirinto")
    solve_p.add_argument("--data", default="maze_data", help="Cartella di training (se serve addestrare)")
    solve_p.add_argument("--epochs", type=int, default=14)
    solve_p.add_argument("--k", type=int, default=None, help="Numero minimo di tesori da raccogliere (default: tutti)")
    solve_p.add_argument("--debug", action="store_true", help="Mostra a video classificazione celle e soluzione")
    solve_p.set_defaults(func=cmd_solve)

    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
