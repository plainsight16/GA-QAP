# Genetic Algorithm for the Quadratic Assignment Problem (QAP)

This project implements a **Genetic Algorithm (GA)** to solve the Quadratic Assignment Problem (QAP).  
It includes both a **matplotlib convergence plot** and a **turtle-based visualizer** for the best solution.

---

## Setup Instructions

1. Clone this repository:
   ```bash
   git clone https://github.com/plainsight16/GA-QAP.git
   cd GA-QAP

2. Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate    # On macOS/Linux
    venv\Scripts\activate       # On Windows
3. Install dependencies:
    ```bash
    pip install -r requirements.txt

## Results

### Graph of Best fitness score vs Avg fitness score
This plot shows how the **best** and **average** fitness scores evolve across generations.

![](/GA_Convergence.png)
---

### Simulation of the Best Solution
The turtle visualizer simulates the **best assignment solution** by drawing facilities and connections.

![](/Simulator.png)
