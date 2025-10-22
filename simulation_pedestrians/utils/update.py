from simulation_pedestrians.utils.social_forces import social_forces
import numpy as np


def update(frame, positions, velocities, v_desired, relax_time, A, B, wall_repulsion, W, L, dt, scat):
    
    if positions.shape[0] == 0 or not np.any(positions[:, 0] < L):
        # Plus de piétons à afficher, arrêter l'animation
        scat.set_offsets(np.empty((0, 2)))
        return scat,

    f = social_forces(positions, velocities, v_desired, relax_time, A, B, wall_repulsion, W)
    velocities += f * dt
    positions += velocities * dt

    # retirer les piétons sortis
    mask = positions[:, 0] < L
    positions = positions[mask]
    velocities = velocities[mask]

    scat.set_offsets(positions)
    return scat,