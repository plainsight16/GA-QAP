import random

def Selection_Function(population, k: int = 3) -> list:
    """Tournament selection (minimization). Returns [chromosome, cost]."""
    if k <= 0: k = 1
    if k > len(population): k = len(population)
    contestants = random.sample(population, k)
    contestants.sort(key=lambda ind: ind[1])   # lower cost = fitter
    return [contestants[0][0][:], contestants[0][1]]
