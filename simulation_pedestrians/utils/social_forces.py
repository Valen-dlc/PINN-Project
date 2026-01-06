import numpy as np


def social_forces(positions, velocities, v_desired, relax_time, A, B, wall_repulsion, W):
    import numpy as np

    N = positions.shape[0]
    forces = np.zeros_like(positions)

    # ============================
    # 1) FONCTION DENSITÉ LOCALE
    # ============================
    def local_density(positions, R=1.0):
        N = positions.shape[0]
        rho = np.zeros(N)
        area = np.pi * R**2

        for i in range(N):
            d = np.linalg.norm(positions - positions[i], axis=1)
            rho[i] = (np.sum(d < R) - 1) / area


        return rho

    # ================================================
    # 2) CONGESTION : vitesse désirée dépend de ρ(i)
    # ================================================
    rho = local_density(positions, R=1.0)  # densité locale autour de chaque piéton

    rho_max = 4.5  # densité critique des piétons (m^-2)
    # loi décroissante : v_effective = v0 * (1 - rho/rho_max)
    v_desired_effective = v_desired * np.exp(-rho / rho_max)


    # ================================================
    # 3) FORCE DÉSIRÉE (modifiée)
    # ================================================
    v0 = np.zeros_like(velocities)
    v0[:, 0] = v_desired_effective   # <-- la modification principale
    #v0[:, 0] = v_desired ancienne vitesse
    forces += (v0 - velocities) / relax_time

    # ================================================
    # 4) RÉPULSION ENTRE PIÉTONS 
    # ================================================
    for i in range(N):
        for j in range(i + 1, N):
            d = positions[i] - positions[j]
            dist = np.linalg.norm(d)
            if dist > 1e-5:
                f = A * np.exp(-dist / B) * (d / dist)
                forces[i] += f
                forces[j] -= f

    # ================================================
    # 5) RÉPULSION AVEC LES MURS 
    # ================================================
    for i in range(N):
        y = positions[i, 1]
        forces[i, 1] += wall_repulsion / (y + 1e-3)        # mur bas
        forces[i, 1] -= wall_repulsion / (W - y + 1e-3)    # mur haut

    return forces




