import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulation_pedestrians.run_simulation import run_pedestrian_simulation
from simulation_pedestrians.utils.calcul_density import calcul_density

if __name__ == "__main__":
    [positions,velocities]=run_pedestrian_simulation(N = 100  ,show=True)
    calcul_density(All_positions=positions,nx=100,ny=28,show=False)

