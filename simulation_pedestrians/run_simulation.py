import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulation_pedestrians.utils.update import update

def run_pedestrian_simulation(L: float = 100.0,
                              W: int = 7,
                              N: int = 15,
                              v_desired: float = 1.3,
                              relax_time: float = 0.5,
                              A: float = 2.0,
                              B: float = 0.5,
                              wall_repulsion: int = 1,
                              dt: float = 0.5,
                              steps: int = 200, 
                              showPlot:bool=True ):

    # --- Initialisation des états ---
    positions = np.zeros((N, 2))
    positions[:, 0] = np.random.uniform(0, L / 4, N)   # départ côté gauche
    positions[:, 1] = np.random.uniform(0, W, N)       # position latérale
    
    velocities = np.zeros((N, 2))

    #Matrice 3D de toutes les positions et vitesses
    all_positions = np.full((steps, N, 2), np.nan)     # NaN par défaut pour tracer proprement
    all_velocities = np.full((steps, N, 2), np.nan)

    # --- Figure / Axes ---
    fig, ax = plt.subplots(figsize=(12, 3))
    scat = ax.scatter(positions[:, 0], positions[:, 1])
    ax.set_xlim(0, L)
    ax.set_ylim(0, W)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Simulation de piétons (N = {N})")


   # --- Animation ---
    def init():
        scat.set_offsets(positions)
        return (scat,)

    if isinstance(showPlot, bool):
        if showPlot:
            ani = FuncAnimation(fig, 
                                update, 
                                frames=steps, 
                                init_func=init,
                                interval=50, #ms
                                blit=True, #Maj que les parties modifiées
                                repeat=False,
                                fargs=(positions, 
                                        velocities, 
                                        v_desired, 
                                        relax_time, 
                                        A, 
                                        B,
                                        wall_repulsion, 
                                        W, 
                                        L, 
                                        dt, 
                                        scat, 
                                        steps, 
                                        all_positions, 
                                        all_velocities))
            plt.show()
        else:
            print("\n /!\ WARNING : 'showPlot' is set to False. The simulation will run without displaying the animation. \n" )
            for step in range(steps):
                update(step, 
                       positions, 
                       velocities, 
                       v_desired, 
                       relax_time, 
                       A, 
                       B,
                       wall_repulsion, 
                       W, 
                       L, 
                       dt, 
                       scat, 
                       steps, 
                       all_positions, 
                       all_velocities)
    else:
        raise ValueError("\nThe 'showPlot' parameter must be a boolean. \n")


    return all_positions, all_velocities

    

if __name__ == "__main__":
    pos, vit = run_pedestrian_simulation(showPlot=True)   

    # print(pos)

    # plt.scatter(pos[100,:, 0], pos[100,:, 1])
    # plt.show()
    




