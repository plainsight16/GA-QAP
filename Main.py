import tkinter as tk
import turtle
from tkinter import ttk
import random, statistics, time
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from utils import *
from graph_visualiser import *
from turtle_visualiser import QAPVisualizer


# -------------------- GA Components --------------------

def fitness(chromosome: list[int]) -> float:
    size = len(chromosome)
    score = 0
    for i in range(size):
        for j in range(size):
            f_i = chromosome[i]
            f_j = chromosome[j]
            score += FLOW_MATRIX[f_i][f_j] * DISTANCE_MATRIX[i][j]
    return score


def pmx_crossover(p1: list[int], p2: list[int]) -> tuple[list[int], list[int]]:
    size = len(p1)
    c1, c2 = [-1] * size, [-1] * size

    start, end = sorted(random.sample(range(size), 2))
    c1[start:end] = p1[start:end]
    c2[start:end] = p2[start:end]

    def fill(child, parentA, parentB):
        for i in range(start, end):
            if parentB[i] not in child:
                pos = i
                val = parentB[i]
                while child[pos] != -1:
                    pos = parentB.index(parentA[pos])
                child[pos] = val
        for i in range(size):
            if child[i] == -1:
                child[i] = parentB[i]
        return child

    fill(c1, p1, p2)
    fill(c2, p2, p1)
    return c1, c2


def swap_mutation(chromosome: list[int]) -> list[int]:
    size = len(chromosome)
    c = chromosome[:]
    i, j = random.sample(range(size), 2)
    c[i], c[j] = c[j], c[i]
    return c


def tournament_selection(population: list[list[int]], k: int = 3) -> list[int]:
    competitors = random.sample(population, k)
    competitors.sort(key=fitness)
    return competitors[0]


def genetic_algorithm(N, POP_SIZE, GENERATIONS, canvas, ax, best_line, avg_line,
                      mutation_rate=0.2, crossover_rate=0.9):
    global FLOW_MATRIX, DISTANCE_MATRIX

    population = generate_population(N, POP_SIZE)
    best = None
    best_score = float('inf')
    best_scores = []
    avg_scores = []
    best_gen = 1

    for gen in range(GENERATIONS):
        new_population = []

        population.sort(key=fitness)
        elite = population[0]
        new_population.append(elite)

        while len(new_population) < POP_SIZE:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)

            if random.random() < crossover_rate:
                child1, child2 = pmx_crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2

            if random.random() < mutation_rate:
                child1 = swap_mutation(child1)

            if random.random() < mutation_rate:
                child2 = swap_mutation(child2)

            new_population.extend([child1, child2])

        population = new_population[:POP_SIZE]

        # Tracking
        population.sort(key=fitness)
        fitness_scores = [fitness(ind) for ind in population]
        score = fitness_scores[0]
        avg = statistics.mean(fitness_scores)
        avg_scores.append(avg)
        best_scores.append(score)

        if score < best_score:
            best_score = score
            best = population[0]
            best_gen = gen

        # ---- LIVE PLOTTING ----
        best_line.set_data(range(1, gen + 2), best_scores)
        avg_line.set_data(range(1, gen + 2), avg_scores)
        ax.relim()
        ax.autoscale_view()
        canvas.draw()
        root.update_idletasks()
        time.sleep(0.05)  # Simulate runtime delay

        print(f"Gen {gen + 1}: Best score = {best_score}, Avg={avg:.2f}")

    print("\nFinal Best Solution:", best, "Score:", best_score)
    return best, best_score, best_scores, avg_scores, best_gen


# -------------------- GUI --------------------

def run_ga():
    global FLOW_MATRIX, DISTANCE_MATRIX

    N = int(n_var.get())
    POP_SIZE = N * 20
    GENERATIONS = 50
    crossover_rate = 0.9
    mutation_rate = 0.2

    DISTANCE_MATRIX = generate_dist_or_flow(N, 100)
    FLOW_MATRIX = generate_dist_or_flow(N, 10)

    # Show matrices
    matrix_display.delete("1.0", tk.END)
    matrix_display.insert(tk.END, f"Distance Matrix (N={N}):\n{DISTANCE_MATRIX}\n\n")
    matrix_display.insert(tk.END, f"Flow Matrix:\n{FLOW_MATRIX}\n")

    # Reset convergence plot
    ax.clear()
    ax.set_title("GA Convergence Curve")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    best_line, = ax.plot([], [], label="Best fitness", color="blue")
    avg_line, = ax.plot([], [], label="Average fitness", color="orange")
    ax.legend()

    # Run GA
    best, best_score, best_scores, avg_scores, best_gen = genetic_algorithm(
        N, POP_SIZE, GENERATIONS, canvas, ax, best_line, avg_line,
        mutation_rate, crossover_rate
    )

    # Final results
    result_label.config(
        text=f"Best Permutation: {best}\nGeneration Number: {best_gen + 1}\nBest Cost: {best_score}"
    )

    # --- Turtle Visualization ---
    screen = turtle.TurtleScreen(turtle_canvas)   # <--- wrap Tkinter canvas
    viz = QAPVisualizer(screen=screen, draw_every=1, flow_pen_scale=1.2)
    viz.draw_generation(best, generation_number=best_gen, best_score=best_score,
                        flow_matrix=FLOW_MATRIX, distance_matrix=DISTANCE_MATRIX)


# Tkinter main window
root = tk.Tk()
root.title("QAP Genetic Algorithm")

# Dropdown for N
ttk.Label(root, text="Select N:").pack()
n_var = tk.StringVar(value="5")
n_dropdown = ttk.Combobox(root, textvariable=n_var, values=[str(i) for i in range(3, 11)])
n_dropdown.pack()

# Run button
run_btn = ttk.Button(root, text="Run GA", command=run_ga)
run_btn.pack()

# Matrices display
matrix_display = tk.Text(root, height=10, width=60)
matrix_display.pack(pady=10)

# Matplotlib figure for convergence plot
fig = Figure(figsize=(6, 4), dpi=100)
ax = fig.add_subplot(111)
ax.set_title("GA Convergence Curve")
ax.set_xlabel("Generation")
ax.set_ylabel("Fitness")
best_line, = ax.plot([], [], label="Best fitness", color="blue")
avg_line, = ax.plot([], [], label="Average fitness", color="orange")
ax.legend()

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Label to show final results
result_label = ttk.Label(root, text="Best solution will appear here after run.")
result_label.pack(pady=10)

# Frame for Turtle visualization
turtle_frame = tk.Frame(root)
turtle_frame.pack(fill=tk.BOTH, expand=True)

# Canvas for turtle inside Tkinter
turtle_canvas = tk.Canvas(turtle_frame, width=400, height=400, bg="white")
turtle_canvas.pack()

root.mainloop()
