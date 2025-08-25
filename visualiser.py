import matplotlib.pyplot as plt

def ga_convergence_plot(best_scores, avg_scores):
    plt.plot(best_scores, label="Best fitness")
    plt.plot(avg_scores, label="Average fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.title("GA Convergence Curve")
    plt.show()