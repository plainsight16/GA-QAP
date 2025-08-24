import random

def Crossover_Function(data1, data2):
    """Permutation-safe OX crossover. Returns two children."""
    p1, p2 = data1[0], data2[0]
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))

    def ox(pa, pb):
        child = [None]*n
        child[a:b] = pa[a:b]
        fill = [g for g in pb if g not in child]
        i = b
        for g in fill:
            if i == n: i = 0
            child[i] = g; i += 1
        return child

    return [[ox(p1, p2), 0], [ox(p2, p1), 0]]
