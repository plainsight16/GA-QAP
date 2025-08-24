def Cost_Function(population, distances: dict, flows: dict) -> list:
    """Compute sum_i sum_j F[i,j] * D[pi(i), pi(j)] for each individual."""
    for ind in population:
        pi = ind[0]; n = len(pi)
        cost = 0
        for i in range(n):
            for j in range(n):
                cost += flows[(i, j)] * distances[(pi[i], pi[j])]
        ind[1] = cost
    return population
