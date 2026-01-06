import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

from simulation_pedestrians.run_simulation import run_pedestrian_simulation
from simulation_pedestrians.utils.calcul_flux import compute_global_fundamental_point


if __name__ == "__main__":
    N_list = [10, 30, 50, 80, 120, 150, 200, 250, 300, 350]

    global_rho = []   # un point par simulation
    global_phi = []

    for N in N_list:
        print(f"\n--- Simulation avec {N} piétons ---")

        # 1. Simulation micro
        positions, velocities = run_pedestrian_simulation(N=N, showPlot=False)
        
        # 2. Diagramme fondamental global → UN SEUL POINT
        rho_G, phi_G = compute_global_fundamental_point(
                            positions, velocities,
                            L=100.0, W=7.0
                        )

        print(f"ρ_global = {rho_G:.4f}  |  φ_global = {phi_G:.4f}")

        # 4. Stocke le point
        global_rho.append(rho_G)
        global_phi.append(phi_G)

    # 5. Trace le diagramme fondamental global (un point par simulation)
    plt.figure(figsize=(8,5))
    plt.scatter(global_rho, global_phi, s=60, c='red')
    plt.xlabel("Densité globale moyenne ρ̄ (piétons/m²)")
    plt.ylabel("Flux global moyen Φ̄ (piétons/s)")
    plt.title("Diagramme fondamental global (un point par simulation)")
    plt.grid(True)
    plt.show()


