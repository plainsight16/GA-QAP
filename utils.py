import numpy as np
import random

def generate_chromosome(length) -> list[int]:
    # location -> facility where location == index and facility == value
    return random.sample(range(length), length)

def generate_population(N, pop_size):
    return [generate_chromosome(N) for _ in range(pop_size)]

def generate_dist_or_flow(size, upper_bound):
    '''
    Generate a N X N matrix for distance or flow
    '''
    matrix = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(i+1, size):
            value = np.random.randint(1, upper_bound)
            matrix[j, i] = value
            matrix[i, j] = value
    return matrix