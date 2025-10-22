import numpy as np


def social_forces(positions, velocities, v_desired, relax_time, A, B, wall_repulsion, W):
    N = positions.shape[0]
    forces = np.zeros_like(positions)

    # Force désirée vers la sortie (droite)
    v0 = np.zeros_like(velocities)
    v0[:, 0] = v_desired
    forces += (v0 - velocities) / relax_time

    # Répulsion entre piétons
    for i in range(N):
        for j in range(i + 1, N):
            d = positions[i] - positions[j]
            dist = np.linalg.norm(d)
            if dist > 1e-5:
                f = A * np.exp(-dist / B) * (d / dist)
                forces[i] += f
                forces[j] -= f

    # Répulsion avec les murs
    for i in range(N):
        y = positions[i, 1]
        forces[i, 1] += wall_repulsion / (y + 1e-3)        # mur bas
        forces[i, 1] -= wall_repulsion / (W - y + 1e-3)    # mur haut

    return forces