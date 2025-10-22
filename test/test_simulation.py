import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulation_pedestrians.run_simulation import run_pedestrian_simulation

if __name__ == "__main__":
    run_pedestrian_simulation(N = 200)

