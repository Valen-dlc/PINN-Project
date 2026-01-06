import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

from simulation_pedestrians.run_simulation import run_pedestrian_simulation
from simulation_pedestrians.utils.calcul_density import calcul_density
from simulation_pedestrians.utils.champ_vitesse import compute_velocity_field

from simulation_pedestrians.utils.calcul_flux import compute_local_fundamental_point


if __name__ == "__main__":

    N_list = [10, 30, 50, 80, 120, 150, 200, 250, 300, 350]

    all_rho = []
    all_phi = []

    for N in N_list:
        print(f"\n--- Simulation avec {N} piétons ---")

        # 1) Simulation micro
        positions, velocities = run_pedestrian_simulation(
            N=N,
            showPlot=False
        )

        # 2) Diagramme fondamental LOCAL (nuage)
        rho, phi = compute_local_fundamental_point(
            positions,
            velocities,
            calcul_density,
            compute_velocity_field,
            L=100.0,
            W=7.0
        )

        print(f"   → {len(rho)} points")

        all_rho.append(rho)
        all_phi.append(phi)

    # 3) Scatter final
    all_rho = np.concatenate(all_rho)
    all_phi = np.concatenate(all_phi)

    plt.figure(figsize=(8,5))
    plt.scatter(all_rho, all_phi, s=5, alpha=0.25)
    plt.xlabel("Densité locale ρ (piétons/m²)")
    plt.ylabel("Flux longitudinal |φₓ| (piétons/m·s)")
    plt.title("Diagramme fondamental microscopique (local)")
    plt.grid(True)
    plt.show()
