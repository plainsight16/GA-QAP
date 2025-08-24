import random

def Generate_Initial_Population(problem_size: int, population_size: int) -> list:
    return [[random.sample(range(problem_size), problem_size), 0]
            for _ in range(population_size)]
