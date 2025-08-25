import statistics
from utils import *
from visualiser import *

def fitness(chromosome: list[int]) -> float:
    size = len(chromosome)
    score = 0
    for i in range(size):
        for j in range(size):
            f_i = chromosome[i]
            f_j = chromosome[j]
            score += FLOW_MATRIX[f_i][f_j] * DISTANCE_MATRIX[i][j]
    return score

def pmx_crossover(p1:list[int], p2:list[int]) -> tuple[list[int], list[int]]:
    size = len(p1)
    c1, c2 = [-1]*size, [-1]*size

    # choose two cut points
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

def genetic_algorithm(N, POP_SIZE, GENERATIONS, mutation_rate=0.2, crossover_rate=0.9):
    population = generate_population(N, POP_SIZE)
    best = None
    best_score = float('inf')
    best_scores = []
    avg_scores = []

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

        # tracking
        population.sort(key=fitness)
        fitness_scores = [fitness(ind) for ind in population]
        score = fitness_scores[0]
        avg = statistics.mean(fitness_scores)
        avg_scores.append(avg)
        best_scores.append(score)
        if score < best_score:
            best_score = score
            best = population[0]

        print(f"Gen {gen+1}: Best score = {best_score}, Avg={avg:.2f}")

    print("\nFinal Best Solution:", best, "Score:", best_score)
    return best, best_score, best_scores, avg_scores


if __name__ == "__main__":
    N = 5
    POP_SIZE = 100
    GENERATIONS = 50
    crossover_rate = 0.9
    mutation_rate = 0.2

    DISTANCE_MATRIX = generate_dist_or_flow(N, 100)
    FLOW_MATRIX = generate_dist_or_flow(N, 10)

    best, best_score, best_scores, avg_scores = genetic_algorithm(N, POP_SIZE, GENERATIONS, mutation_rate, crossover_rate)
    ga_convergence_plot(best_scores, avg_scores)