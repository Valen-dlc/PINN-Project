import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
from simulation_pedestrians.run_simulation import run_pedestrian_simulation
from simulation_pedestrians.utils.calcul_density import calcul_density
from simulation_pedestrians.utils.champ_vitesse import animate_velocity_field_grid
from simulation_pedestrians.utils.champ_vitesse import compute_velocity_field
from simulation_pedestrians.utils.calcul_flux import compute_and_plot_fundamental

if __name__ == "__main__":
    N_list = [10, 30, 50, 80, 120, 150, 200, 250]

    all_rho = []
    all_phi = []

    for N in N_list:
        print(f"--- Simulation avec {N} piétons ---")

        # 1. Simulation micro
        positions, velocities = run_pedestrian_simulation(N=N, showPlot=False)

        # 2. Densité
        D = calcul_density(all_positions=positions, nx=50, ny=7,N=N, 
                        showPlot=False, normalize=False)

        # 3. Fondamental : récupère rho et phi depuis TA fonction
        rho, phi, U, V, phi_norm = compute_and_plot_fundamental(
                                    D, positions, velocities,
                                    compute_velocity_field,
                                    nx=50, ny=7)

        # 4. Sauvegarde dans les listes globales
        all_rho.extend(rho)
        all_phi.extend(phi)

    # 5. Diagramme fondamental global
    plt.figure(figsize=(8,5))
    plt.scatter(all_rho, all_phi, s=4, alpha=0.3)
    plt.xlabel("Densité ρ")
    plt.ylabel("Flux ||φ||")
    plt.title("Diagramme fondamental consolidé (plusieurs N)")
    plt.grid(True)
    plt.show()




    #positions, velocities = run_pedestrian_simulation(N = 150, showPlot=False)
    #print(velocities.size)

    #D=calcul_density(all_positions=positions,nx=500,ny=49,showPlot=True, normalize=False)
    #animate_velocity_field_grid(positions,velocities,nx=500,ny=49)
    #compute_and_plot_fundamental(D,positions,velocities,compute_velocity_field,nx=500,ny=49)

