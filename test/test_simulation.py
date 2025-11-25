import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulation_pedestrians.run_simulation import run_pedestrian_simulation
from simulation_pedestrians.utils.calcul_density import calcul_density
from simulation_pedestrians.utils.champ_vitesse import animate_velocity_field_grid
from simulation_pedestrians.utils.champ_vitesse import compute_velocity_field
from simulation_pedestrians.utils.calcul_flux import compute_and_plot_fundamental

if __name__ == "__main__":
    positions, velocities = run_pedestrian_simulation(N = 150, showPlot=False)
    print(velocities.size)

    D=calcul_density(all_positions=positions,nx=500,ny=49,showPlot=True, normalize=False)
    animate_velocity_field_grid(positions,velocities,nx=500,ny=49)
    compute_and_plot_fundamental(D,positions,velocities,compute_velocity_field,nx=500,ny=49)

