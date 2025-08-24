import random

def Mutation_Function(data) -> list:
    """Swap two positions in a permutation chromosome."""
    chrom = data[0]
    if len(chrom) >= 2:
        i, j = random.sample(range(len(chrom)), 2)
        chrom[i], chrom[j] = chrom[j], chrom[i]
    return data
