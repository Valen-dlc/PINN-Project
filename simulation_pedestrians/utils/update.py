from simulation_pedestrians.utils.social_forces import social_forces
import numpy as np


def update2(frame,
           positions, velocities, v_desired, relax_time, A, B, wall_repulsion, W, L, dt, scat,
           steps, All_positions, All_velocities):

    # Si plus rien à afficher
    if positions.shape[0] == 0 or not np.any(positions[:, 0] < L):
        scat.set_offsets(np.empty((0, 2)))
        return (scat,)

    # Intégration
    f = social_forces(positions, velocities, v_desired, relax_time, A, B, wall_repulsion, W)
    velocities += f * dt
    positions  += velocities * dt

    # Enregistrement à l’index de frame donné par Matplotlib
    if frame < steps:
        n = positions.shape[0]
        All_positions[frame, :n, :] = positions
        All_positions[frame, n:,  :] = np.nan
        All_velocities[frame, :n, :] = velocities
        All_velocities[frame, n:,  :] = np.nan

    # Affiche seulement les piétons encore dans la zone, sans réassigner l’objet positions
    mask = positions[:, 0] < L
    scat.set_offsets(positions[mask])

    return (scat,)