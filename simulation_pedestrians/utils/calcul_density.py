import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#voir si sa marche 

from simulation_pedestrians.utils.update import update_density



def calcul_density(All_positions,L: float = 150.0, W: int = 7,N: int = 15,nx: int=100, ny:int = 14,
                            dt: float = 0.5,
                            steps: int = 200, show:bool=True):
    # --- Tracé des trajectoires enregistrées ---
    fig2, ax2 = plt.subplots(figsize=(12, 3))
    ax2.set_xlim(0, L)
    ax2.set_ylim(0, W)
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.set_title("Trajectoires des piétons")

    subset = np.linspace(0, N - 1, min(N, 20), dtype=int)  # max 20 traj. pour la lisibilité
    for i in subset:
        traj = All_positions[:, i, :]
        ax2.plot(traj[:, 0], traj[:, 1], alpha=0.3)
    if show :
     plt.show()
    # ==== paramètres du cadrillage ====
    nx, ny = 100, 14      # nb de cases en x et y (ajuste si besoin)
    normalize =False    # True: densité normalisée par nb de piétons actifs à chaque frame

    # ==== bords des cellules ====
    x_edges = np.linspace(0, L, nx+1)
    y_edges = np.linspace(0, W, ny+1)
    x_cent = 0.5*(x_edges[:-1] + x_edges[1:])
    y_cent = 0.5*(y_edges[:-1] + y_edges[1:])

    steps = All_positions.shape[0]
    D = np.zeros((steps, ny, nx), dtype=float)   # densité par frame

   # ==== calcule la densité par frame ====
    for t in range(steps):
     pos_t = All_positions[t]                 # (N, 2)
     mask = np.isfinite(pos_t[:,0]) & np.isfinite(pos_t[:,1])
     x = pos_t[mask, 0]
     y = pos_t[mask, 1]
     H, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges])  # (ny, nx)

     if normalize:
         H = H / max(1, mask.sum())          # normalisation par nb de piétons présents
     D[t] = H


    # ==== 2) animation frame par frame ====
    fig, ax = plt.subplots(figsize=(12, 3))
    quad = ax.pcolormesh(x_edges, y_edges, D[0], shading='auto')
    cbar = plt.colorbar(quad, ax=ax, label=("densité normalisée" if normalize else "comptes"))
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_xlim(0, L); ax.set_ylim(0, W)
    ax.set_title("Densité par pas de temps")

    # échelle de couleur fixe pour éviter le pompage (prend des percentiles robustes)
    vmin = np.percentile(D, 5)
    vmax = np.percentile(D, 95)
    quad.set_clim(vmin, vmax)
    
    if show:
     ani2 = FuncAnimation(
        fig, update_density, frames=steps,
        interval=50, blit=False, repeat=False,
        fargs=(quad,D,steps,ax))
     plt.tight_layout(); plt.show()

    return (D)

