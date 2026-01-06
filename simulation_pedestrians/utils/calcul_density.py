import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from simulation_pedestrians.utils.update import update_density


def calcul_density(all_positions,
                   L: float = 100.0, 
                   W: int = 7,
                   nx: int = 100, 
                   ny: int = 14,
                   normalize=False,
                   showPlot: bool = True):

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    # ====================================================
    # NOMBRE RÉEL DE PIÉTONS
    # ====================================================
    N_real = all_positions.shape[1]

    # --- Tracé des trajectoires enregistrées ---
    fig2, ax2 = plt.subplots(figsize=(12, 3))
    ax2.set_xlim(0, L)
    ax2.set_ylim(0, W)
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.set_title("Trajectoires des piétons")

    subset = np.linspace(0, N_real - 1, min(N_real, 20), dtype=int)
    for i in subset:
        traj = all_positions[:, i, :]
        ax2.plot(traj[:, 0], traj[:, 1], alpha=0.3)

    if showPlot:
        plt.show()

    # ====================================================
    # PARAMÈTRES DU CADRILLAGE
    # ====================================================
    dx = L / nx
    dy = W / ny
    cell_area = dx * dy

    x_edges = np.linspace(0, L, nx + 1)
    y_edges = np.linspace(0, W, ny + 1)

    steps = all_positions.shape[0]
    D = np.zeros((steps, ny, nx), dtype=float)

    # ====================================================
    # CALCUL DE LA DENSITÉ
    # ====================================================
    for t in range(steps):
        pos_t = all_positions[t]
        mask = np.isfinite(pos_t[:, 0]) & np.isfinite(pos_t[:, 1])
        x = pos_t[mask, 0]
        y = pos_t[mask, 1]

        H, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges])

        if normalize:
            H = H / max(1, mask.sum())

        D[t] = H / cell_area

    return D

