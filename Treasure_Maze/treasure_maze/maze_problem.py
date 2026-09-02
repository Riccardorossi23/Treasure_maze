# -*- coding: utf-8 -*-
"""
Definizione del problema Treasure Maze come problema di ricerca AIMA.

Rispetto alla versione originale, qui:
- esiste UNA SOLA classe TreasureMaze (nel file originale ne comparivano
  quattro definizioni diverse, una sovrascritta dall'altra: due erano
  "scheletri" con metodi placeholder mai completati e mai usati, e la loro
  sola presenza rendeva il modulo enormemente più difficile da leggere e
  mantenere. Restava attiva solo l'ultima, le altre erano codice morto).
- lo stato include ESPLICITAMENTE i tesori già raccolti, così la ricerca
  può davvero risolvere l'obiettivo "raccogli tutti i tesori" o "raccogline
  almeno k", come richiesto dal progetto (capitolo 1.1 del pdf). La
  versione originale, invece, lanciava una ricerca A*/UCS separata verso
  ogni tesoro preso in ordine di lista: questo non è un vero "problema di
  ricerca con più obiettivi", non garantisce l'ottimalità del percorso
  complessivo e non permette affatto la modalità "almeno k tesori".
- i muri ('X') sono ATTRAVERSABILI abbattendoli a costo 5, come descritto
  nel dominio (capitolo 1, "X: muro, abbattibile con costo 5"). Nel codice
  originale i muri erano invece celle proibite (rimosse da actions()),
  in contraddizione con la consegna.
- il conteggio delle "celle esplorate" viene fatto correttamente
  instrumentando il problema (aima.search.InstrumentedProblem), invece di
  usare per errore la lunghezza del percorso soluzione (bug presente nel
  codice originale: veniva chiamata num_visited_cells la lunghezza di
  astar_solution.path(), che è il percorso finale, non l'insieme dei nodi
  espansi durante la ricerca).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Tuple

from aima.search import Problem, InstrumentedProblem, astar_search, uniform_cost_search

Position = Tuple[int, int]
State = Tuple[Position, FrozenSet[Position], FrozenSet[Position]]
# state = (posizione_corrente, tesori_raccolti, muri_abbattuti)

WALL = "X"
START = "S"
TREASURE = "T"
WALL_COST = 5


class TreasureMaze(Problem):
    """Problema di ricerca: un agente si muove su una griglia NxM per
    raccogliere tesori ('T'), potendo attraversare muri ('X') abbattendoli
    a un costo fisso. Ogni cella calpestabile ha un costo di transito
    numerico (1-4). L'obiettivo è raccogliere tutti i tesori, oppure
    almeno `k` di essi se `k` è specificato.
    """

    def __init__(self, maze: List[List[str]], start: Position, k: Optional[int] = None):
        self.maze = maze
        self.rows = len(maze)
        self.cols = len(maze[0]) if self.rows else 0
        self.treasures = frozenset(
            (r, c) for r in range(self.rows) for c in range(self.cols) if maze[r][c] == TREASURE
        )
        if not self.treasures:
            raise ValueError("Nessun tesoro ('T') presente nel labirinto.")
        if k is not None and not (1 <= k <= len(self.treasures)):
            raise ValueError(f"k deve essere compreso tra 1 e {len(self.treasures)}.")
        self.k = k  # None -> raccogliere tutti i tesori

        initial: State = (start, frozenset(), frozenset())
        super().__init__(initial)

    # -- utilità -----------------------------------------------------
    def _in_bounds(self, pos: Position) -> bool:
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _cell_cost(self, pos: Position, broken_walls: FrozenSet[Position]) -> int:
        r, c = pos
        value = self.maze[r][c]
        if value == WALL:
            return 1 if pos in broken_walls else WALL_COST
        if value.isdigit():
            return int(value)
        return 1  # 'S', 'T' o cella vuota: costo base

    # -- interfaccia Problem ------------------------------------------
    def actions(self, state: State) -> List[str]:
        (row, col), _, _ = state
        moves = {
            "UP": (row - 1, col),
            "DOWN": (row + 1, col),
            "LEFT": (row, col - 1),
            "RIGHT": (row, col + 1),
        }
        return [name for name, pos in moves.items() if self._in_bounds(pos)]

    def result(self, state: State, action: str) -> State:
        (row, col), collected, broken = state
        delta = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}[action]
        new_pos = (row + delta[0], col + delta[1])

        if self.maze[new_pos[0]][new_pos[1]] == WALL:
            broken = broken | {new_pos}

        if new_pos in self.treasures:
            collected = collected | {new_pos}

        return new_pos, collected, broken

    def goal_test(self, state: State) -> bool:
        _, collected, _ = state
        needed = self.k if self.k is not None else len(self.treasures)
        return len(collected) >= needed

    def path_cost(self, c, state1: State, action: str, state2: State) -> int:
        (_, collected1, broken1) = state1
        new_pos, _, _ = state2
        return c + self._cell_cost(new_pos, broken1)

    def h(self, node) -> int:
        """Euristica ammissibile: distanza di Manhattan dal tesoro
        non ancora raccolto più vicino (0 se l'obiettivo è già
        soddisfatto)."""
        (row, col), collected, _ = node.state
        remaining = self.treasures - collected
        if not remaining:
            return 0
        return min(abs(row - tr) + abs(col - tc) for tr, tc in remaining)


@dataclass
class SearchResult:
    algorithm: str
    path: Optional[List[Position]]
    cost: Optional[int]
    time_seconds: float
    states_generated: int  # celle (stati) effettivamente generati durante la ricerca
    goal_tests: int


def _run(maze: List[List[str]], start: Position, k: Optional[int], algorithm: str) -> SearchResult:
    problem = TreasureMaze(maze, start, k=k)
    instrumented = InstrumentedProblem(problem)

    t0 = time.perf_counter()
    if algorithm == "astar":
        node = astar_search(instrumented, instrumented.h)
    elif algorithm == "ucs":
        node = uniform_cost_search(instrumented)
    else:
        raise ValueError("algorithm deve essere 'astar' o 'ucs'")
    elapsed = time.perf_counter() - t0

    if node is None:
        return SearchResult(algorithm, None, None, elapsed, instrumented.states, instrumented.goal_tests)

    path = [s[0] for s in [n.state for n in node.path()]]
    return SearchResult(algorithm, path, node.path_cost, elapsed, instrumented.states, instrumented.goal_tests)


def solve_treasure_maze(
    maze: List[List[str]],
    start: Optional[Position] = None,
    k: Optional[int] = None,
) -> Tuple[SearchResult, SearchResult]:
    """Risolve il labirinto con A* e UCS e restituisce i due risultati,
    inclusi tempo di esecuzione e numero di stati generati (celle
    esplorate), per il confronto di prestazioni richiesto nel capitolo 3
    del progetto.
    """
    if start is None:
        start = find_start(maze)
    astar_result = _run(maze, start, k, "astar")
    ucs_result = _run(maze, start, k, "ucs")
    return astar_result, ucs_result


def find_start(maze: List[List[str]]) -> Position:
    for r, row in enumerate(maze):
        for c, value in enumerate(row):
            if value == START:
                return r, c
    raise ValueError("Nessuna posizione di partenza ('S') trovata nel labirinto.")
