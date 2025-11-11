import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulation_pedestrians.run_simulation import run_pedestrian_simulation
from simulation_pedestrians.utils.calcul_density import calcul_density

if __name__ == "__main__":
    positions, velocities = run_pedestrian_simulation(N = 50, showPlot=True)

    calcul_density(all_positions=positions,nx=50,ny=4,showPlot=True, normalize=False)

