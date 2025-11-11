import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from simulation_pedestrians.utils.social_forces import social_forces


def update(frame,
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
           all_velocities):

    # Si plus rien à afficher
    if positions.shape[0] == 0 or not np.any(positions[:, 0] < L):
        scat.set_offsets(np.empty((0, 2)))
        plt.close()
        print("\n > Simulation completed \n > The graphics window has been closed \n")
        return (scat,)

    # Intégration
    f = social_forces(positions, velocities, v_desired, relax_time, A, B, wall_repulsion, W)
    velocities += f * dt
    positions  += velocities * dt

    # Enregistrement à l’index de frame donné par Matplotlib
    if frame < steps:
        n = positions.shape[0]
        all_positions[frame, :n, :] = positions
        all_positions[frame, n:,  :] = np.nan
        all_velocities[frame, :n, :] = velocities
        all_velocities[frame, n:,  :] = np.nan

    # Affiche seulement les piétons encore dans la zone, sans réassigner l’objet positions
    mask = positions[:, 0] < L
    scat.set_offsets(positions[mask])

    return (scat,)


def update_density(k,quad,D,steps,ax):
    quad.set_array(D[k].ravel())  # pcolormesh attend un array aplati (ny*nx,)
    ax.set_title(f"Densité au pas t={k+1}/{steps}")
    return (quad,)

