import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def compute_velocity_field(All_positions, All_velocities, L, W, nx=100, ny=14):
    """
    Calcule la vitesse moyenne (U,V) par cellule de la grille.
    Retourne: U, V de shape (steps, ny, nx)
    """
    T, N, _ = All_positions.shape

    x_edges = np.linspace(0, L, nx+1)
    y_edges = np.linspace(0, W, ny+1)

    U = np.zeros((T, ny, nx))
    V = np.zeros((T, ny, nx))

    for t in range(T):
        pos_t = All_positions[t]
        vel_t = All_velocities[t]

        mask = np.isfinite(pos_t[:,0]) & np.isfinite(pos_t[:,1]) & np.isfinite(vel_t[:,0]) & np.isfinite(vel_t[:,1])
        x = pos_t[mask,0]; y = pos_t[mask,1]
        u = vel_t[mask,0]; v = vel_t[mask,1]

        # on récupère l’indice du bin pour chaque agent
        ix = np.clip(np.digitize(x, x_edges)-1, 0, nx-1)
        iy = np.clip(np.digitize(y, y_edges)-1, 0, ny-1)

        # sommes et comptages pour chaque cellule
        sum_u = np.zeros((ny, nx))
        sum_v = np.zeros((ny, nx))
        count = np.zeros((ny, nx))

        np.add.at(sum_u, (iy, ix), u)
        np.add.at(sum_v, (iy, ix), v)
        np.add.at(count, (iy, ix), 1)

        # moyenne par cellule (évite division par 0)
        with np.errstate(invalid='ignore', divide='ignore'):
            U[t] = np.where(count>0, sum_u/count, 0)
            V[t] = np.where(count>0, sum_v/count, 0)

    return U, V


def animate_velocity_field_grid(All_positions, All_velocities,nx=100, ny=14, L: float = 100.0,
                              W: int = 7,
                                fps=10, scale_quiver=1.0, show=True):
    """
    Affiche/retourne une animation de la vitesse moyenne par cellule de grille.
    - Quiver: vecteur vitesse moyen (U,V)
    - Heatmap: norme |v| moyenne

    Retourne: (U, V, ani) où
      U,V : arrays (T, ny, nx) de vitesses moyennes par cellule
      ani  : l'objet FuncAnimation (utile si tu veux sauvegarder)
    """
    # --- centres de cellules ---
    x_edges = np.linspace(0, L, nx+1)
    y_edges = np.linspace(0, W, ny+1)
    x_cent = 0.5*(x_edges[:-1] + x_edges[1:])
    y_cent = 0.5*(y_edges[:-1] + y_edges[1:])
    Xc, Yc = np.meshgrid(x_cent, y_cent, indexing='xy')  # (ny, nx)

    # --- calcule U,V (vitesse moyenne sur grille) ---
    U, V = compute_velocity_field(All_positions, All_velocities, L, W, nx=nx, ny=ny)
    T = U.shape[0]
    speed = np.hypot(U, V)  # norme |v|

    # bornes robustes pour l’échelle de couleur
    vmin = np.nanpercentile(speed, 5)
    vmax = np.nanpercentile(speed, 95)
    if not np.isfinite(vmin): vmin = 0.0
    if not np.isfinite(vmax): vmax = 1.0

    # --- figure ---
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_xlim(0, L); ax.set_ylim(0, W)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Vitesse moyenne par cellule (quiver) + |v| (fond)")

    # fond: plus simple avec imshow que pcolormesh pour mise à jour
    im = ax.imshow(speed[0], origin='lower', extent=(0, L, 0, W),
                   vmin=vmin, vmax=vmax, interpolation='nearest', aspect='auto')
    cbar = plt.colorbar(im, ax=ax, label="|v| moyenne (m/s)")

    # quiver: initialisation
    # éviter division par zéro pour 'scale'
    s = (1.0/scale_quiver) if scale_quiver not in (0, None) else 1.0
    quiv = ax.quiver(Xc, Yc, U[0], V[0], angles='xy', scale_units='xy',
                     scale=s, width=0.003)

    def update(i):
        im.set_data(speed[i])      # (ny, nx)
        quiv.set_UVC(U[i], V[i])   # met à jour les vecteurs
        return im, quiv

    ani = FuncAnimation(fig, update, frames=T, interval=int(1000//fps),
                        blit=False, repeat=False)

    if show:
        plt.tight_layout()
        plt.show()

    return U, V, ani
