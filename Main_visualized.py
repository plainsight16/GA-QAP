
# Main_visualized.py
"""
Visualized GA runner for the Quadratic Assignment Problem (QAP).
Uses your existing GA modules plus a turtle-based visualizer.

Files expected alongside this script:
  - GeneratePopulation.py  (Generate_Initial_Population)
  - Selection.py           (Selection_Function)
  - Crossover.py           (Crossover_Function)
  - Mutation.py            (Mutation_Function)
  - Fitness.py             (Cost_Function)
  - utils.py               (Generate_Distance_Or_Flow OR similar)
  - qap_visualizer.py      (QAPVisualizer)
"""
import random
import numpy as np

from GeneratePopulation import Generate_Initial_Population
from Selection import Selection_Function
from Crossover import Crossover_Function
from Mutation import Mutation_Function
from Fitness import Cost_Function
from utils import Generate_Distance_Or_Flow

from qap_visualizer import QAPVisualizer


def dict_to_matrix(d: dict, n: int) -> np.ndarray:
    """Convert utils.Generate_Distance_Or_Flow() dict to an NxN numpy array."""
    M = np.zeros((n, n), dtype=float)
    for (i, j), v in d.items():
        M[i, j] = v
    return M


def run_ga_qap_visual(
    n: int = 9,
    pop_size: int = 40,
    iters: int = 200,
    pc: float = 0.9,
    pm: float = 0.2,
    k: int = 3,
    elite_frac: float = 0.1,
    draw_every: int = 5,             # draw every N generations
    sleep_seconds: float = 0.3,  # seconds to pause after each draw (0 = no pause)
    draw_on_improvement: bool = True,  # also draw when we find a new best
    seed: None = None,            # random seed for repeatability
):
    if seed is not None:
        random.seed(seed)

    # Problem instance (dicts from utils) + numpy copies for visualization
    D_dict = Generate_Distance_Or_Flow(n, 100)
    F_dict = Generate_Distance_Or_Flow(n, 10)
    D = dict_to_matrix(D_dict, n)
    F = dict_to_matrix(F_dict, n)

    # Initial population & fitness
    population = Generate_Initial_Population(n, pop_size)   # [[perm, 0], ...]
    population = Cost_Function(population, D_dict, F_dict)  # your fitness function uses dicts

    # Best-so-far
    best = min(population, key=lambda ind: ind[1])
    best = [best[0][:], best[1]]

    # Visualizer
    viz = QAPVisualizer(draw_every=draw_every, flow_pen_scale=1.2, sleep_seconds=sleep_seconds)
    # Initial draw
    viz.draw_generation(best[0], generation_number=0, best_score=best[1], flow_matrix=F, distance_matrix=D)

    # --- Evolution loop ---
    for gen in range(1, iters + 1):
        # Elitism
        elites = sorted(population, key=lambda ind: ind[1])[:max(1, int(elite_frac * pop_size))]
        next_pop: list[list] = [[e[0][:], e[1]] for e in elites]

        # Fill remainder
        while len(next_pop) < pop_size:
            p1 = Selection_Function(population)  # [perm, cost]
            p2 = Selection_Function(population)

            if random.random() < pc:
                children = Crossover_Function(p1, p2)  # [[perm, cost],[perm, cost]]
            else:
                children = [[p1[0][:], 0], [p2[0][:], 0]]

            for c in children:
                if random.random() < pm:
                    Mutation_Function(c)
                next_pop.append(c)
                if len(next_pop) >= pop_size:
                    break

        # Evaluate
        population = Cost_Function(next_pop, D_dict, F_dict)
        cur_best = min(population, key=lambda ind: ind[1])

        # Update global best
        if cur_best[1] < best[1]:
            best = [cur_best[0][:], cur_best[1]]
            if draw_on_improvement:
                viz.draw_generation(best[0], generation_number=gen, best_score=best[1], flow_matrix=F, distance_matrix=D)

        # Periodic draw
        if (gen % draw_every) == 0:
            viz.draw_generation(cur_best[0], generation_number=gen, best_score=cur_best[1], flow_matrix=F, distance_matrix=D)

    print("Best cost:", best[1])
    print("Best assignment:", best[0])
    viz.hold()
    return best


if __name__ == "__main__":
    run_ga_qap_visual()
